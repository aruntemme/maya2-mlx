"""Maya-2 text-to-speech on MLX, one stream at a time.

The per-frame logic is maya3.engine.Engine's for a single seat: prompt plan (frame caps, minimum, guard windows),
controlled stop (tau 0.35 / hazard 0.6 / argmax AEOS) + cb0 sampling with temperature / top-k / top-p / presence
penalty, the 31-level depth decoder, the streaming Mimi decode, the loop guard with its EOS boost, and a silence gap
between text pieces. Text normalisation, splitting, prompt packing, voices and guards reuse the repo's own code.
Reference-audio cloning encodes the clip with the PyTorch Mimi encoder and ECAPA on CPU (once per voice).
"""
import random
import time

import mlx.core as mx
import numpy as np

# maya3.engine puts the release's root and scripts/ on sys.path on import, so tts.* and scripts.* resolve after it
from maya3 import guards as G
from maya3.engine import EngineConfig, FRAME, REASONS, ROOT, SR, _check_sampling
from maya3.prompt import DEFAULT_BUILDER, MODES, Voice, VoiceRegistry, load_named_voices, norm_mode

from . import weights as W
from .backbone import Backbone
from .depth import Depth
from .mimi import Mimi
from .prompt import Composer, prompt_batch

AEOS_CLASS = 2048


class Maya2MLX:
    def __init__(self, paths, config: EngineConfig = None, backbone_bits=None):
        from scripts import s2_serve_controls as S
        from tts.data.packing_v2 import PackerV2
        self.S = S
        self.cfg = cfg = config or EngineConfig(max_seats=1)
        self.paths = paths
        t0 = time.perf_counter()
        w = W.load(paths["ckpt"])
        self.backbone = Backbone(w, paths["backbone_dir"], capacity=cfg.capacity, bits=backbone_bits)
        self.composer = Composer(w)
        self.depth = Depth(w, dtype={"bf16": mx.bfloat16, "fp16": mx.float16, "fp32": mx.float32}[cfg.depth_dtype],
                           temp=cfg.temp, top_k=cfg.top_k)
        self.mimi = Mimi(paths["mimi_dir"])
        tok, ids = W.tokenizer(paths["backbone_dir"])
        self.packer = PackerV2(tok, ids, max_tokens=8192)
        self.prior = S.load_prior(str(ROOT / "configs/text_frames_prior.json"))
        self.dur = G.DurationPrior()
        self.registry = VoiceRegistry()
        self._seed_rng = random.Random(cfg.seed)
        self._oneoff = {}
        self._encoder = self._ecapa = None
        mx.eval(self.depth.embed)
        self.load_s = time.perf_counter() - t0

    def warmup(self):
        """Trace the compiled depth frame and touch every kernel once, so the first request starts at full speed."""
        t0 = time.perf_counter()
        for _ in self._piece(self.prepare_voice(self.registry.get("plain")), "Hello.", "en", {"seed": 1}, 0, 0, 3):
            pass
        self.warmup_s = time.perf_counter() - t0
        return self.warmup_s

    @classmethod
    def from_pretrained(cls, repo="maya-research/maya2-native", voices=True, backbone_bits=None, warmup=True, **config):
        """config: EngineConfig fields (e.g. depth_dtype="bf16"); backbone_bits: None (bf16) | 8 | 4."""
        paths = W.resolve(repo)
        tts = cls(paths, EngineConfig(**{"max_seats": 1, **config}), backbone_bits=backbone_bits)
        if warmup:
            tts.warmup()
        if voices and paths.get("voices_json"):
            load_named_voices(tts.registry, paths["voices_json"])
        return tts

    # ------------------------------------------------------------------ voices
    @property
    def voices(self):
        return self.registry.ids()

    def prepare_voice(self, v: Voice):
        """Prefix tokens, speaker memory and the prefix backbone KV, once per voice (Engine.prepare_voice)."""
        if getattr(v, "mlx", None) is not None:
            return v
        use_icl, use_mem = MODES[v.mode]
        builder = v.builder or DEFAULT_BUILDER
        v.prefix = builder.prefix(self.packer, v)
        v.spk_refs = np.asarray(v.ecapa, np.float32).reshape(1, 192) if v.ecapa is not None else np.zeros((1, 192), np.float32)
        v.ts = int(v.prefix["tokens"].shape[0]) if v.prefix is not None else 0
        cache = {"cross": self.depth.speaker(v.ecapa if use_mem else None), "kv": None}
        if v.ts:
            ex = builder.example(self.packer, v, "a", v.ref_lang or "en")
            emb = self.composer.compose(prompt_batch(ex)[0])[:v.ts]
            self.backbone.reset()
            self.backbone.forward(emb)
            cache["kv"] = self.backbone.prefix_kv(v.ts)
            mx.eval(cache["kv"])
        mx.eval(cache["cross"])
        v.mlx = cache
        return v

    def _encode_reference(self, w):
        import torch
        from transformers import MimiModel
        from maya3.api import Ecapa
        if self._encoder is None:
            self._encoder = MimiModel.from_pretrained(self.paths["mimi_dir"]).eval()
        with torch.inference_mode():
            codes = self._encoder.encode(torch.as_tensor(w)[None, None], num_quantizers=32).audio_codes[0].numpy()
        if self._ecapa is None:
            self._ecapa = Ecapa(self.paths["ecapa_dir"], "cpu")
        return codes.astype(np.int64), self._ecapa(w)

    def make_voice(self, voice_id, ref_audio=None, ref_text=None, lang=None, mode="icl_spk", meta=None):
        from maya3.api import load_audio
        mode = norm_mode(mode)
        clips, ecapa = [], None
        if ref_audio is not None:
            w = load_audio(ref_audio)
            if len(w) > 30 * SR:
                raise ValueError("reference audio longer than 30 s; 3-10 s works best")
            codes, emb = self._encode_reference(w)
            if mode in ("spk", "icl_spk"):
                ecapa = emb
            if mode in ("icl", "icl_spk"):
                if not ref_text or not lang:
                    raise ValueError("ref_text and lang (the reference clip's transcript and language) are required for cloning")
                clips = [{"text": ref_text, "codes": codes, "lang": lang}]
        return self.prepare_voice(Voice(voice_id, mode, clips=clips, ecapa=ecapa, meta=meta or {}))

    def register_voice(self, voice_id, ref_audio=None, ref_text=None, lang=None, mode="icl_spk", **kw):
        self.registry.add(self.make_voice(voice_id, ref_audio, ref_text, lang, mode, **kw))
        return voice_id

    def _voice(self, voice, ref_audio, ref_text, ref_lang, mode):
        if ref_audio is None:
            return self.prepare_voice(self.registry.get(voice or "plain"))
        if voice is not None and voice != "plain":
            raise ValueError("pass either voice=<registered/named voice> or ref_audio=, not both")
        key = (str(ref_audio), ref_text, ref_lang, mode)
        if key not in self._oneoff:
            self._oneoff[key] = self.make_voice(f"oneoff-{len(self._oneoff)}", ref_audio, ref_text, ref_lang, mode)
        return self._oneoff[key]

    # ------------------------------------------------------------------ synthesis
    def _plan(self, voice, text, lang, frames_done, max_frames):
        S, cfg, g = self.S, self.cfg, self.cfg.guards
        ex = (voice.builder or DEFAULT_BUILDER).example(self.packer, voice, text, lang)
        pb, n = prompt_batch(ex)
        ef, er = S.expected_frames(self.prior, lang, len(text))
        loose = cap = min(cfg.hard_cap, int(2.0 * (ef + er) + 40))
        if g.enabled:
            cap = min(cap, max(int(ef * 0.5) + 8, self.dur.cap_frames(text, lang, g.dur_k, g.dur_margin_s)))
        if max_frames:
            cap = min(cap, max(1, int(max_frames) - frames_done))
        cap = min(cap, cfg.capacity - n - 2)
        minimum = max(4, int(0.5 * ef))
        if g.enabled and g.loop and g.aware_v2:
            suspend = G.suspend_windows_v2(text, lang, self.dur)
        else:
            suspend = G.tag_windows(text, lang, self.dur, g.tag_hold_s) if (g.enabled and g.loop and g.tag_aware and "[" in text) else ()
        return self.composer.compose(pb), n, cap, loose, minimum, suspend

    def _cb0(self, h, st, key):
        """Engine._cb0 for one seat -> (stop code [1], cb0 [1], new hazard)."""
        logits = self.backbone.cb0_logits(h).astype(mx.float32) / st["temp"]
        logits = logits + st["bias"] * (mx.arange(AEOS_CLASS + 1) == AEOS_CLASS)
        if self.cfg.cb0_rep_win > 0 and st["rep"] > 0 and st["recent"]:
            seen = np.zeros(AEOS_CLASS + 1, np.float32)
            seen[list(st["recent"])] = 1.0
            seen[AEOS_CLASS] = 0
            logits = logits - st["rep"] * mx.array(seen)
        kth = mx.topk(logits, st["top_k"], -1).min(-1, keepdims=True)
        logits = mx.where(logits < kth, -mx.inf, logits)
        if st["top_p"] < 1.0:
            order = mx.argsort(-logits, -1)
            sp = mx.take_along_axis(mx.softmax(logits, -1), order, -1)
            drop_sorted = (mx.cumsum(sp, -1) - sp) > st["top_p"]
            drop = mx.put_along_axis(mx.zeros_like(drop_sorted), order, drop_sorted, -1)
            logits = mx.where(drop, -mx.inf, logits)
        probs = mx.softmax(logits, -1)
        pa = probs[:, AEOS_CLASS]
        can = mx.array(st["frames"] >= st["minimum"])
        hz = mx.where(can, 1 - (1 - st["hazard"]) * (1 - pa), st["hazard"])
        code = mx.zeros_like(pa, dtype=mx.int32)
        code = mx.where(can & (hz >= .6), 4, code)
        code = mx.where(can & (pa >= .35), 3, code)
        code = mx.where(can & (mx.argmax(probs, -1) == AEOS_CLASS), 2, code)
        q = probs * (mx.arange(AEOS_CLASS + 1) != AEOS_CLASS)
        q = q / mx.maximum(q.sum(-1, keepdims=True), 1e-12)
        race = -mx.log(mx.random.uniform(shape=q.shape, key=key))
        cb0 = mx.minimum(mx.argmax(q / race, -1), AEOS_CLASS - 1)
        return code, cb0, hz

    def _piece(self, voice, text, lang, sampling, piece_idx, frames_done, max_frames):
        """Yield int16 PCM frames for one text piece; returns (frames, stop reason) via StopIteration.value."""
        cfg, g = self.cfg, self.cfg.guards
        emb, n, cap, loose, minimum, suspend = self._plan(voice, text, lang, frames_done, max_frames)
        if cap < 1:
            return 0, "capacity"
        if sampling.get("seed") is not None:
            seed = (int(sampling["seed"]) * 1000003 + piece_idx * 7919) % (2 ** 31 - 1)
        else:
            seed = self._seed_rng.randrange(1, 2 ** 31 - 1)
        key = mx.random.key(seed)
        st = {"temp": float(sampling.get("temperature") or cfg.cb0_temp), "top_k": int(sampling.get("top_k") or cfg.cb0_top_k),
              "top_p": float(sampling.get("top_p") or cfg.cb0_top_p),
              "rep": float(sampling["repetition_penalty"] if sampling.get("repetition_penalty") is not None else cfg.cb0_rep),
              "bias": 0.0, "recent": [], "frames": 0, "minimum": minimum, "hazard": mx.zeros((1,), mx.float32)}
        cross = voice.mlx["cross"]
        bb = self.backbone
        bb.reset()
        if voice.ts:
            bb.load_prefix(voice.mlx["kv"], voice.ts)
            h = bb.forward(emb[voice.ts:])[-1:]
        else:
            h = bb.forward(emb)[-1:]
        self.mimi.reset()
        hist = np.full(g.hist, -1, np.int64)
        frames, boost_at, reason = 0, None, None
        while True:
            key, k_cb0, k_depth = mx.random.split(key, 3)
            code, cb0, st["hazard"] = self._cb0(h, st, k_cb0)
            codes, fb = self.depth(h, cb0, cross, k_depth)
            pcm = mx.round(mx.clip(self.mimi.decode(codes[0]), -1, 1) * 32767).astype(mx.int16)
            h = bb.forward(fb)          # next frame's hidden state, in the same evaluation (discarded if this frame stops)
            mx.eval(code, cb0, pcm, h, st["hazard"])
            stop = int(code.item())
            st["frames"] += 1
            if stop:
                reason = REASONS[stop]
                break
            c0 = int(cb0.item())
            frames += 1
            yield np.array(pcm)
            hist[:-1] = hist[1:]
            hist[-1] = c0
            if cfg.cb0_rep_win > 0:
                st["recent"] = (st["recent"] + [c0])[-cfg.cb0_rep_win:]
            if frames >= cap:
                reason = "max_frames" if max_frames and frames_done + frames >= max_frames else \
                    ("cap_guard" if g.enabled and cap < loose else "cap")
                break
            if g.enabled and g.loop and frames >= g.start and frames % g.every == 0 and not any(a <= frames <= b for a, b in suspend):
                hit = G.loop_detect_batch(hist[None], np.array([frames]), g)[0]
                if hit is not None and g.mode == "boost":
                    if boost_at is None:
                        boost_at, st["bias"] = frames, g.boost
                    if frames - boost_at >= g.backstop:
                        reason = "loop_backstop"
                        break
                elif hit is not None:
                    reason = "loop_" + hit.split(":")[0]
                    break
                elif boost_at is not None:
                    boost_at, st["bias"] = None, 0.0
        return frames, reason

    def stream(self, text, lang="en", voice=None, ref_audio=None, ref_text=None, ref_lang=None, mode="icl_spk",
               max_frames=None, normalize=False, info=None, seed=None, temperature=None, top_k=None, top_p=None,
               repetition_penalty=None):
        """Yield int16 PCM chunks (24 kHz mono, one 80 ms frame each) as they are generated."""
        sampling = _check_sampling({"seed": seed, "temperature": temperature, "top_k": top_k, "top_p": top_p,
                                    "repetition_penalty": repetition_penalty})
        v = self._voice(voice, ref_audio, ref_text, ref_lang or lang, mode)
        if normalize:
            from maya3.text import normalise_text
            text = normalise_text(text, lang)
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")
        text = text.strip()
        pieces = self.S.split_text(text, lang, self.prior, self.cfg.chunk_s) if self.cfg.chunk_s > 0 else [text]
        t0, t_first, total, stops = time.perf_counter(), None, 0, []
        for i, piece in enumerate(pieces):
            gen = self._piece(v, piece, lang, sampling, i, total, max_frames)
            while True:
                try:
                    pcm = next(gen)
                except StopIteration as e:
                    n, reason = e.value
                    break
                t_first = t_first or time.perf_counter()
                yield pcm
            total += n
            stops.append({"reason": reason, "frames": n})
            if max_frames and total >= max_frames:
                break
            if i + 1 < len(pieces):
                yield np.zeros(int(self.cfg.gap_s * SR), np.int16)
        self.last_info = {"reason": stops[-1]["reason"] if stops else None, "frames": total, "audio_s": total * FRAME / SR,
                          "ttfa_s": (t_first - t0) if t_first else None, "wall_s": time.perf_counter() - t0,
                          "pieces": len(pieces), "stops": stops}
        if info is not None:
            info.update(self.last_info)

    def generate(self, text, lang="en", voice=None, return_info=False, **kw):
        """Synthesise text -> float32 waveform (24 kHz)."""
        info = {}
        chunks = list(self.stream(text, lang, voice, info=info, **kw))
        wav = (np.concatenate(chunks).astype(np.float32) / 32767.0) if chunks else np.zeros(0, np.float32)
        return (wav, info) if return_info else wav

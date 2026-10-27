"""Prompt building: the repo's own packer/collate (CPU, no model) + compose_inputs in MLX.

prompt_batch() is scripts/s2_serve_controls.prompt_from_example without its final model.compose_inputs call, so the
token / span geometry comes from the training code unchanged. compose() is TTSModelV2.compose_inputs
(text embedding, audio-frame embedding, AEOS input, speaker anchor at every span's <|audio|> sentinel).
"""
import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np

KIND_AUDIO, KIND_AEOS = 1, 3
# torch nn.RMSNorm(eps=None) on bf16 input computes in fp32 with fp32's epsilon (measured: bf16's 2**-7 is 50% off)
RMS_EPS = float(np.finfo(np.float32).eps)


def prompt_batch(ex, q_serve=32):
    """-> (numpy batch dict cut at the target's first audio column, n_prompt_tokens)."""
    from tts.data.packing_v2 import collate_v2
    from tts.model.load_s3 import truncate_prompt_batch

    b = collate_v2([ex], q_choices=(q_serve,), q_rng=np.random.default_rng(0))
    kind = b["kind"][0]
    ts = int(ex.get("target_start", 0))
    a = (kind == KIND_AUDIO).nonzero().flatten()
    a = a[a >= ts]
    assert a.numel(), "prompt has no target audio position"
    a0 = int(a[0])
    pb = truncate_prompt_batch(b, a0, require_target_span=True)
    return {k: (v.numpy() if hasattr(v, "numpy") else v) for k, v in pb.items()}, a0


class Composer:
    def __init__(self, w, n_codes=2048):
        self.embed = w["backbone.lm.model.embed_tokens.weight"]
        self.table = w["audio_in.table.weight"]
        self.norm = w["audio_in.norm.weight"]
        self.aeos_in = w["audio_in.aeos_in"]
        self.fc1_w, self.fc1_b = w["speaker_anchor.fc1.weight"], w["speaker_anchor.fc1.bias"]
        self.fc2_w, self.fc2_b = w["speaker_anchor.fc2.weight"], w["speaker_anchor.fc2.bias"]
        self.null_vec = w["speaker_anchor.null_vec"]
        self.n_codes = n_codes
        self.eps = RMS_EPS

    def frames(self, codes, q_use):
        """codes [N, >=q_use] int -> [N, D]: sum of per-level table rows / sqrt(q), RMS-normed."""
        lvl = mx.arange(q_use) * self.n_codes
        e = self.table[mx.array(codes[:, :q_use]) + lvl].sum(1) / math.sqrt(q_use)   # bf16 sum and divide, as torch
        return mx.fast.rms_norm(e, self.norm, self.eps)

    def anchor(self, pb):
        refs = mx.array(pb["spk_refs"][0, :, :192], mx.float32)
        m = pb.get("spk_ref_mask")
        if m is None:
            pooled = refs.mean(0)
            n_real = refs.shape[0]
        else:
            mm = mx.array(m[0], mx.float32)[:, None]
            pooled = (refs * mm).sum(0) / mx.maximum(mm.sum(), 1.0)
            n_real = int(m[0].sum())
        null = bool(np.asarray(pb.get("spk_null", [False]))[0]) or n_real == 0
        x = (self.null_vec if null else pooled).astype(self.fc1_w.dtype)
        return nn.gelu(x @ self.fc1_w.T + self.fc1_b) @ self.fc2_w.T + self.fc2_b

    def compose(self, pb):
        """-> [T, D] prompt embeddings (bf16), exactly TTSModelV2.compose_inputs for one row."""
        tokens, kind, codes = pb["tokens"][0], pb["kind"][0], pb["codes"][0]
        q_use = int(pb["q_use"])
        emb = self.embed[mx.array(tokens)]
        audio = np.nonzero(kind == KIND_AUDIO)[0]
        if audio.size:
            emb[mx.array(audio)] = self.frames(codes[audio], q_use).astype(emb.dtype)
        aeos = np.nonzero(kind == KIND_AEOS)[0]
        if aeos.size:
            emb[mx.array(aeos)] = mx.fast.rms_norm(self.aeos_in, self.norm, self.eps).astype(emb.dtype)
        cols = np.asarray(pb["span_sentinel"])[np.asarray(pb["span_batch_row"]) == 0]
        if cols.size:
            a = self.anchor(pb).astype(emb.dtype)
            for c in cols.tolist():      # index_put(accumulate=True): one add per sentinel
                emb[c] = emb[c] + a
        return emb

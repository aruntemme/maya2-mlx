"""Parity: prompt embeddings, backbone prefill + 16 decode steps, cb0 logits (MLX vs the PyTorch model on MPS).

Yardstick: torch's own bf16 noise, measured as torch full-prefill vs torch two-stage prefill of the same prompt.
"""
import numpy as np
import torch
import mlx.core as mx

import ref
from maya2_mlx import weights as W
from maya2_mlx.backbone import Backbone
from maya2_mlx.prompt import Composer, prompt_batch
from scripts import s2_serve_controls as S
from tts.data.packing_v2 import PackerV2

torch.manual_seed(0)
r = ref.paths()
model = ref.torch_model(r)
tok, ids = W.tokenizer(r["backbone_dir"])
assert len(tok) == len(model.backbone.tokenizer) and ids == model.backbone.token_ids, "tokenizer mismatch"
packer = PackerV2(tok, ids, max_tokens=8192)
reg = ref.voices(r, packer)
w = W.load(r["ckpt"])
comp, bb = Composer(w), Backbone(w, r["backbone_dir"])

f = lambda t: t.float().cpu().numpy()
from maya3.prompt import Voice
rng = np.random.default_rng(1)
e = rng.standard_normal(192).astype(np.float32)
clone = Voice("clone", "icl_spk", ecapa=e / np.linalg.norm(e),   # prompt math only: random reference codes are fine here
              clips=[{"text": "This is my reference sentence.", "lang": "en", "codes": rng.integers(0, 2048, (32, 40))}])
reg.add(ref.prepare_host(clone, packer))
cases = [("clone", "hi", "मुझे यह आवाज़ बहुत पसंद है।"),
         ("Kayal", "ta", "வேற லெவல்ல இருக்கு, நாளைக்கு மீட்டிங் மறக்காதே!"),
         ("plain", "en", "Your order has shipped and will arrive tomorrow."),
         ("Ritvik", "hi", "नमस्ते, आपका ऑर्डर कल पहुँच जाएगा।")]
with torch.inference_mode():
    for vid, lang, text in cases:
        v = reg.get(vid)
        ex = (v.builder or __import__("maya3.prompt").prompt.DEFAULT_BUILDER).example(packer, v, text, lang)
        pt, n_t = S.prompt_from_example(ex, model, ref.DEV, torch.bfloat16, 32)
        pt = pt.to(torch.bfloat16)
        pb, n_m = prompt_batch(ex)
        pm = comp.compose(pb)
        mx.eval(pm)
        e_t, e_m = f(pt[0]), np.array(pm.astype(mx.float32))
        print(f"\n[{vid}/{lang}] prompt tokens torch {n_t} mlx {n_m} | embeddings: exact {np.mean(e_t == e_m):.4f}, "
              f"max|d| {np.abs(e_t - e_m).max():.2e}, rel {ref.rel(e_m, e_t):.2e}")

        # torch: full prefill, plus a two-stage prefill as the bf16 noise yardstick
        h_t, cache = model.backbone(pt, use_cache=True)
        split = n_t // 2
        h_a, c2 = model.backbone(pt[:, :split], use_cache=True)
        h_b, _ = model.backbone(pt[:, split:], past_key_values=c2, use_cache=True)
        bb.reset()
        h_m = bb.forward(pm)
        lt, lm = model.backbone.cb0_logits(h_t[0, -1]).float(), bb.cb0_logits(h_m[-1]).astype(mx.float32)
        mx.eval(h_m, lm)
        lt, lm = lt.cpu().numpy(), np.array(lm)
        pt_, pm_ = np.exp(lt - lt.max()) / np.exp(lt - lt.max()).sum(), np.exp(lm - lm.max()) / np.exp(lm - lm.max()).sum()
        kl = float((pt_ * (np.log(pt_ + 1e-12) - np.log(pm_ + 1e-12))).sum())
        print(f"  prefill last hidden: rel {ref.rel(np.array(h_m[-1].astype(mx.float32)), f(h_t[0, -1])):.2e} "
              f"(torch noise floor {ref.rel(f(h_b[0, -1]), f(h_t[0, -1])):.2e}) | cb0 argmax equal "
              f"{lt.argmax() == lm.argmax()}, top-64 overlap {len(set(np.argsort(-lt)[:64]) & set(np.argsort(-lm)[:64]))}/64, KL {kl:.2e}")

        # 16 decode steps with identical feedback embeddings (random frames through the audio input table)
        errs, floor = [], []
        for step in range(16):
            codes = torch.randint(0, 2048, (1, 32), device=ref.DEV)
            x_t = model.audio_in.frames(codes, 32).to(torch.bfloat16)
            x_m = mx.array(f(x_t)).astype(mx.bfloat16)
            h_t, cache = model.backbone(x_t[None], past_key_values=cache, use_cache=True)
            h_m = bb.forward(x_m)
            mx.eval(h_m)
            errs.append(ref.rel(np.array(h_m[-1].astype(mx.float32)), f(h_t[0, -1])))
        print(f"  decode x16 hidden rel: mean {np.mean(errs):.2e}, max {np.max(errs):.2e}")

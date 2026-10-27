"""Parity: depth decoder (MLX fp16) vs the fp32 reference DepthV2 and the engine's FastDepth (fp16), on real backbone states."""
import copy

import numpy as np
import torch
import mlx.core as mx

import ref
from maya2_mlx import weights as W
from maya2_mlx.depth import Depth, GROUP_OF_LEVEL
from serve.fast_depth import FastDepth
from scripts import s2_serve_controls as S
from tts.data.packing_v2 import PackerV2
from maya3.prompt import DEFAULT_BUILDER

r = ref.paths()
model = ref.torch_model(r)
packer = PackerV2(model.backbone.tokenizer, model.backbone.token_ids, max_tokens=8192)
reg = ref.voices(r, packer)
model.depth.float()
d32 = model.depth
d16 = copy.deepcopy(d32).half()
fast = FastDepth(model, 1, dtype=torch.float16, greedy=True, fused=False)
w = W.load(r["ckpt"])
dm = Depth(w)
f = lambda t: t.float().cpu().numpy()
rng = np.random.default_rng(0)
ecapa = rng.standard_normal(192).astype(np.float32)
ecapa /= np.linalg.norm(ecapa)

with torch.inference_mode():
    v = reg.get("Kayal")
    ex = DEFAULT_BUILDER.example(packer, v, "வேற லெவல்ல இருக்கு, நாளைக்கு மீட்டிங் மறக்காதே!", "ta") if v.builder is None \
        else v.builder.example(packer, v, "வேற லெவல்ல இருக்கு, நாளைக்கு மீட்டிங் மறக்காதே!", "ta")
    pt, _ = S.prompt_from_example(ex, model, ref.DEV, torch.bfloat16, 32)
    for label, refs in (("nulled memory (named voice)", None), ("ECAPA memory (clone path)", ecapa)):
        refs_t = torch.as_tensor(refs if refs is not None else np.zeros(192, np.float32), device=ref.DEV)[None, None]
        mem32 = d32.spk_memory(refs_t, "ecapa", null_mask=torch.tensor([refs is None], device=ref.DEV))
        rmask = torch.ones(1, 1, dtype=torch.bool, device=ref.DEV)
        fast.set_speaker(0, mem32, rmask)
        cross = dm.speaker(refs)
        h_all, cache = model.backbone(pt.to(torch.bfloat16), use_cache=True)
        h = h_all[:, -1]
        rel_m, rel_t, agree, first_miss, fb_rel = [], [], [], [], []
        for frame in range(12):
            l = model.backbone.cb0_logits(h).float()
            cb0 = l[:, :2048].argmax(-1)
            codes_t, fb_t = fast([0], h, cb0)                       # engine fp16, greedy
            # teacher-forced logits on the engine's codes: fp32 reference, torch fp16, MLX fp16
            hm, cm, fm = mx.array(f(h)).astype(mx.bfloat16), mx.array(cb0.cpu().numpy()), mx.array(codes_t.cpu().numpy())
            _, fb_m, lg_m = dm._compute(hm, cm, cross, None, force=fm)
            codes_m, _, _ = dm._compute(hm, cm, cross, None)        # MLX greedy
            mx.eval(fb_m, codes_m, *lg_m)
            for level in range(1, 32):
                g = GROUP_OF_LEVEL[level]
                x32 = d32._prefix(h.float(), codes_t, level)
                l32 = d32.heads[level - 1](d32._run(x32, g, mem32, rmask)).float() / 0.7
                x16 = d16._prefix(h.half(), codes_t, level)
                l16 = d16.heads[level - 1](d16._run(x16, g, mem32.half(), rmask)).float() / 0.7
                rel_m.append(ref.rel(np.array(lg_m[level - 1]), f(l32)))
                rel_t.append(ref.rel(f(l16), f(l32)))
            ct, cmn = codes_t.cpu().numpy()[0], np.array(codes_m)[0]
            agree.append(np.mean(ct == cmn))
            miss = np.nonzero(ct != cmn)[0]
            first_miss.append(int(miss[0]) if miss.size else 32)
            fb_rel.append(ref.rel(np.array(fb_m.astype(mx.float32)), f(fb_t)))
            h, cache = model.backbone(fb_t.to(torch.bfloat16)[None], past_key_values=cache, use_cache=True)
            h = h[:, -1]
        print(f"\n[{label}] 12 frames x 31 levels")
        print(f"  teacher-forced logits vs fp32 reference: MLX fp16 rel mean {np.mean(rel_m):.2e} max {np.max(rel_m):.2e} | "
              f"torch fp16 rel mean {np.mean(rel_t):.2e} max {np.max(rel_t):.2e}")
        print(f"  greedy codes equal to engine FastDepth: {np.mean(agree):.3f} of levels; first mismatch level per frame: {first_miss}")
        print(f"  feedback embedding rel (same codes): mean {np.mean(fb_rel):.2e}")

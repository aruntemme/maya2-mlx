"""Accuracy cost of the opt-in speed variants, on real states from a bf16 generation trajectory.

depth dtype: teacher-forced logits (all 31 levels) vs an fp32 MLX depth, for fp16 (the engine default) and bf16.
backbone bits: cb0 distribution KL / argmax agreement vs the bf16 backbone over a prompt + 40 decode frames,
with the bf16-vs-PyTorch KL measured in check_backbone.py (~1e-3) as the yardstick.
"""
import numpy as np
import mlx.core as mx

from maya2_mlx import weights as W
from maya2_mlx.backbone import Backbone
from maya2_mlx.depth import Depth
from maya2_mlx.tts import Maya2MLX

tts = Maya2MLX.from_pretrained(warmup=False)
w = W.load(tts.paths["ckpt"])
v = tts.prepare_voice(tts.registry.get("Kayal"))
text = "இன்று வானிலை நன்றாக உள்ளது. நாம் மாலை கடற்கரைக்கு போகலாம்."
emb = tts._plan(v, text, "ta", 0, None)[0]

# a real trajectory from the reference configuration: hidden states, cb0, codes, feedback
bb, dp = tts.backbone, tts.depth
bb.reset()
h = bb.forward(emb)[-1:]
key = mx.random.key(3)
traj = []
for _ in range(40):
    key, k1, k2 = mx.random.split(key, 3)
    l = bb.cb0_logits(h).astype(mx.float32)
    cb0 = mx.argmax(l[:, :2048] + mx.random.gumbel(shape=(1, 2048), key=k1), -1)
    codes, fb = dp(h, cb0, v.mlx["cross"], k2)
    traj.append((h, cb0, codes, fb))
    h = bb.forward(fb)[-1:]
mx.eval(traj)

# depth precision
d = {dt: Depth(w, dtype=getattr(mx, dt)) for dt in ("float32", "float16", "bfloat16")}
cross = {dt: m.speaker(None) for dt, m in d.items()}
for dt in ("float16", "bfloat16"):
    rels = []
    for h, cb0, codes, _ in traj:
        ref = d["float32"]._compute(h, cb0, cross["float32"], None, force=codes)[2]
        got = d[dt]._compute(h, cb0, cross[dt], None, force=codes)[2]
        for a, b in zip(got, ref):
            a, b = np.array(a), np.array(b)
            rels.append(np.linalg.norm(a - b) / np.linalg.norm(b))
    print(f"depth {dt:8s}: teacher-forced logits rel vs fp32  mean {np.mean(rels):.2e}  max {np.max(rels):.2e}")

# backbone quantization
def cb0_dists(b):
    b.reset()
    out = [b.cb0_logits(b.forward(emb)[-1:])]
    for _, _, _, fb in traj[:-1]:
        out.append(b.cb0_logits(b.forward(fb)[-1:]))
    return [np.array(mx.softmax(o.astype(mx.float32), -1))[0] for o in out]
base = cb0_dists(Backbone(w, tts.paths["backbone_dir"]))
for bits in (8, 4):
    q = cb0_dists(Backbone(w, tts.paths["backbone_dir"], bits=bits))
    kl = [float((p * (np.log(p + 1e-12) - np.log(r + 1e-12))).sum()) for p, r in zip(base, q)]
    agree = np.mean([p.argmax() == r.argmax() for p, r in zip(base, q)])
    print(f"backbone {bits}-bit: cb0 KL vs bf16 mean {np.mean(kl):.2e} max {np.max(kl):.2e} | argmax agreement {agree:.2f}")

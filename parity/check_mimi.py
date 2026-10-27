"""Parity: streaming Mimi decode, MLX vs the engine's StaticMimi (fp32), on real codes from an encoded speech clip."""
import numpy as np
import soundfile as sf
import torch
import mlx.core as mx

import ref
from maya2_mlx.mimi import Mimi
from serve.fast_codec import StaticMimi
from transformers import MimiModel

r = ref.paths()
hf = MimiModel.from_pretrained(r["mimi_dir"]).to(ref.DEV).eval()
from pathlib import Path
wav, sr = sf.read(str(Path(r["ckpt"]) / "samples/en_siddhant.wav"), dtype="float32")   # the release's own English sample
assert sr == 24000
with torch.inference_mode():
    codes = hf.encode(torch.as_tensor(wav, device=ref.DEV)[None, None], num_quantizers=32).audio_codes[0]   # [32, F]
    full = hf.decode(codes[None]).audio_values[0, 0].float().cpu().numpy()
    static = StaticMimi(hf, 1, batch_buckets=(1,))
    static.warmup()
    static.reset(0, zero_kv=True)
    ref_frames = [static.decode([0], codes[:, t][None])[0].float().cpu().numpy() for t in range(codes.shape[1])]
mm = Mimi(r["mimi_dir"])
c_np = codes.cpu().numpy()
mlx_frames = []
for t in range(c_np.shape[1]):
    y = mm.decode(mx.array(c_np[:, t]))
    mx.eval(y)
    mlx_frames.append(np.array(y))
a, b = np.concatenate(mlx_frames), np.concatenate(ref_frames)
snr = lambda x, y: 10 * np.log10((y ** 2).sum() / ((x - y) ** 2).sum())
print(f"{codes.shape[1]} frames, {len(a)} samples")
print(f"MLX vs StaticMimi (engine): rel {ref.rel(a, b):.2e}, SNR {snr(a, b):.1f} dB, max|d| {np.abs(a - b).max():.2e}")
print(f"StaticMimi vs HF full decode: rel {ref.rel(b, full[:len(b)]):.2e} (reference streaming-vs-batch floor)")
print(f"MLX vs HF full decode: rel {ref.rel(a, full[:len(a)]):.2e}")

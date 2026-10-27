"""Transcribe every clip in out/ with Whisper (mlx) and compare to the input text (character error rate).
Run with the separate .venv-asr env:  .venv-asr/bin/python transcribe.py"""
import json
import sys
import unicodedata
from pathlib import Path

import mlx_whisper
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

OUT = Path(__file__).resolve().parents[1] / (sys.argv[1] if len(sys.argv) > 1 else "out")
MODEL = "mlx-community/whisper-large-v3-turbo"


def norm(s):
    s = unicodedata.normalize("NFC", s).lower()
    return "".join(c for c in s if unicodedata.category(c)[0] in "LMN")   # letters, marks, digits only


def cer(ref, hyp):
    r, h = norm(ref), norm(hyp)
    d = list(range(len(h) + 1))
    for i, a in enumerate(r, 1):
        prev, d[0] = d[0], i
        for j, b in enumerate(h, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (a != b))
    return d[len(h)] / max(len(r), 1)


res = json.loads((OUT / "results.json").read_text())
items = dict(res["cases"])
if "stream" in res:
    items["stream_en"] = {"lang": "en", **res["stream"]}
for name, r in items.items():
    wav, sr = sf.read(OUT / f"{name}.wav", dtype="float32")   # 24 kHz -> Whisper's 16 kHz, no ffmpeg needed
    audio = resample_poly(wav, 2, 3).astype(np.float32)
    hyp = mlx_whisper.transcribe(audio, path_or_hf_repo=MODEL, language=r["lang"])["text"].strip()
    r["asr"] = hyp
    r["cer"] = None if name == "en_numbers" else round(cer(r["text"], hyp), 3)   # normalizer expands digits to words
    print(f"{name:15s} cer={r['cer']}  asr={hyp}", flush=True)
    if name in res["cases"]:
        res["cases"][name] = r
    else:
        res["stream"] = {k: v for k, v in r.items() if k != "lang"}
(OUT / "results.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))

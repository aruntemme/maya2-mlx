"""Compare Deepgram Nova-3 language modes (and Whisper turbo) on the bilingual caller lines: transcript + time.

    python dg_compare.py [wav ...]      # defaults to caller_lines/chat/*.wav
"""
import glob
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent


def key():
    if os.environ.get("DEEPGRAM_API_KEY"):
        return os.environ["DEEPGRAM_API_KEY"]
    for line in (HERE.parent / ".env").read_text().splitlines():
        if line.startswith("DEEPGRAM_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("DEEPGRAM_API_KEY not set (environment or the repo .env)")


def deepgram(path, language, k):
    url = f"https://api.deepgram.com/v1/listen?model=nova-3&language={language}&punctuate=true&smart_format=true"
    req = urllib.request.Request(url, Path(path).read_bytes(), {"Authorization": f"Token {k}", "Content-Type": "audio/wav"})
    t = time.perf_counter()
    res = json.load(urllib.request.urlopen(req, timeout=30))
    alt = res["results"]["channels"][0]["alternatives"][0]
    langs = sorted({w.get("language", "") for w in alt.get("words", [])} - {""})
    return time.perf_counter() - t, alt["transcript"], langs


if __name__ == "__main__":
    k = key()
    files = sys.argv[1:] or sorted(glob.glob(str(HERE / "caller_lines/chat/*.wav")))
    for f in files:
        print(f"\n{Path(f).name}")
        for lang in ("multi", "ta", "en-IN"):
            try:
                dt, text, langs = deepgram(f, lang, k)
                print(f"  {lang:6s} {dt*1000:5.0f} ms  {text}  {langs if langs else ''}")
            except Exception as e:
                print(f"  {lang:6s} ERROR {str(e)[:120]}")

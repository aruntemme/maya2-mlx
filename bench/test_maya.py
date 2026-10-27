"""Smoke + quality test of Maya-2 Native on Apple Silicon (MPS).

Generates speech in several languages with preset voices, streams one request, clones a voice
cross-lingually, and writes every clip plus results.json to out/. Transcription check: transcribe.py.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from maya3 import Maya3TTS

SR = 24000
OUT = Path(__file__).resolve().parents[1] / "out"
OUT.mkdir(exist_ok=True)
DEVICE = sys.argv[1] if len(sys.argv) > 1 else "mps"

CASES = [  # (name, lang, voice, text)
    ("en", "en", "Siddhant", "Your order has shipped and will arrive tomorrow before noon."),
    ("en_numbers", "en", "Ahana", "Please pay Rs. 1,250 by 15/11/2026; call 98450 12345 if you have questions."),
    ("hi", "hi", "Ritvik", "नमस्ते, आपका ऑर्डर कल पहुँच जाएगा। धन्यवाद!"),
    ("ta", "ta", "Abhay", "வணக்கம், நீங்கள் எப்படி இருக்கிறீர்கள்? இன்று வானிலை நன்றாக உள்ளது."),
    ("te", "te", "Nivedita", "నమస్కారం, మీరు ఎలా ఉన్నారు? ఈ రోజు వాతావరణం చాలా బాగుంది."),
    ("bn", "bn", "Jahnavi", "নমস্কার, আপনি কেমন আছেন? আজকের আবহাওয়া খুব সুন্দর।"),
    ("kn", "kn", "Satvik", "ನಮಸ್ಕಾರ, ನೀವು ಹೇಗಿದ್ದೀರಿ? ಇಂದು ಹವಾಮಾನ ತುಂಬಾ ಚೆನ್ನಾಗಿದೆ."),
    ("mr", "mr", "Charvi", "नमस्कार, तुम्ही कसे आहात? आज हवामान खूप छान आहे."),
]


def stats(wav):
    wav = np.asarray(wav, np.float32)
    return {"audio_s": round(len(wav) / SR, 2), "peak": round(float(np.abs(wav).max()), 3),
            "rms_db": round(float(20 * np.log10(np.sqrt(np.mean(wav ** 2)) + 1e-9)), 1),
            "finite": bool(np.isfinite(wav).all())}


def main():
    res = {"device": DEVICE, "torch": torch.__version__, "cases": {}}
    t = time.perf_counter()
    tts = Maya3TTS.from_pretrained("maya-research/maya2-native", device=DEVICE, max_seats=1)
    res["load_s"] = round(time.perf_counter() - t, 1)
    voices = sorted(tts.voices)
    res["voices"] = voices
    print(f"loaded in {res['load_s']} s; {len(voices)} voices: {voices}", flush=True)
    try:
        for name, lang, voice, text in CASES:
            if voice not in tts.voices:
                voice = None
            t = time.perf_counter()
            wav = tts.generate(text, lang=lang, voice=voice, normalize=True, seed=7)
            wall = time.perf_counter() - t
            sf.write(OUT / f"{name}.wav", wav, SR)
            r = {"lang": lang, "voice": voice, "text": text, "wall_s": round(wall, 2), **stats(wav)}
            r["rtf"] = round(wall / max(r["audio_s"], 1e-6), 2)
            res["cases"][name] = r
            print(name, r, flush=True)

        # streaming: time to first audio chunk
        text = "Streaming lets playback start before the whole sentence has been generated."
        t = time.perf_counter()
        chunks, ttfa = [], None
        for c in tts.stream(text, lang="en", voice="Siddhant", normalize=True):
            ttfa = ttfa or time.perf_counter() - t
            chunks.append(np.asarray(c))
        pcm = np.concatenate(chunks)
        sf.write(OUT / "stream_en.wav", pcm.astype(np.int16), SR)
        res["stream"] = {"text": text, "chunks": len(chunks), "dtype": str(chunks[0].dtype), "ttfa_s": round(ttfa, 2),
                         "wall_s": round(time.perf_counter() - t, 2), "audio_s": round(len(pcm) / SR, 2)}
        print("stream", res["stream"], flush=True)

        # cross-lingual cloning: English reference (known transcript) -> Hindi output
        ref = OUT / "en.wav"
        tts.register_voice("my_voice", ref_audio=str(ref), ref_text=CASES[0][3], lang="en")
        text = "मुझे यह आवाज़ बहुत पसंद है, और यह हिंदी भी बोल सकती है।"
        t = time.perf_counter()
        wav = tts.generate(text, lang="hi", voice="my_voice", normalize=True, seed=7)
        sf.write(OUT / "clone_en_to_hi.wav", wav, SR)
        res["cases"]["clone_en_to_hi"] = {"lang": "hi", "voice": "my_voice (ref en.wav)", "text": text,
                                          "wall_s": round(time.perf_counter() - t, 2), **stats(wav)}
        print("clone", res["cases"]["clone_en_to_hi"], flush=True)
    finally:
        tts.close()
        (OUT / "results.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()

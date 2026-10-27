"""A real-time voice call agent on a Mac: LiveKit Agents + Deepgram (or local Whisper) + any OpenAI-compatible LLM +
Maya-2 (MLX). The persona, Kayal, speaks casual spoken Tamil by default and English when the caller speaks English.

    python agent.py console   # talk with your Mac's mic and speakers, no server needed
    python agent.py dev       # join rooms on a LiveKit server (LIVEKIT_URL / LIVEKIT_API_KEY / LIVEKIT_API_SECRET)

The LLM is any OpenAI-compatible chat API: LLM_BASE_URL, LLM_API_KEY, LLM_MODEL, optional LLM_REASONING_EFFORT
(default: a local mlx_lm.server on port 18090). Settings and keys come from the environment, else from the repo's
.env, else from a file named by EXTRA_ENV_FILE; see .env.example. Without a DEEPGRAM_API_KEY, speech recognition
falls back to local Whisper. Per-turn latency (ChatMessage.metrics) is appended to turns.jsonl.
"""
import asyncio
import concurrent.futures
import json
import math
import os
import re
import threading
import time
from pathlib import Path

import numpy as np
from livekit import rtc
from livekit.agents import (DEFAULT_API_CONNECT_OPTIONS, Agent, AgentSession, APIConnectOptions, JobContext,
                            JobExecutorType, JobProcess, WorkerOptions, cli, stt, tts, utils)
from livekit.agents.inference.eot import TurnDetector
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.plugins import deepgram, openai, silero   # plugins must register on the main thread, at import
from scipy.signal import resample_poly



def _secret(name):
    """A credential from the environment, else the repo's .env, else the file named by EXTRA_ENV_FILE (never logged)."""
    if os.environ.get(name):
        return os.environ[name]
    files = [Path(__file__).resolve().parents[1] / ".env"]
    if os.environ.get("EXTRA_ENV_FILE"):
        files.append(Path(os.environ["EXTRA_ENV_FILE"]).expanduser())
    for f in files:
        if f.exists():
            for line in f.read_text().splitlines():
                k, sep, v = line.partition("=")
                if sep and k.strip() == name and v.strip():
                    return v.strip().strip('"').strip("'")
    return None


LLM_URL = _secret("LLM_BASE_URL") or "http://127.0.0.1:18090/v1"          # default: local mlx_lm.server
LLM_MODEL = _secret("LLM_MODEL") or "mlx-community/Qwen3-4B-Instruct-2507-4bit"
LLM_REASONING_EFFORT = _secret("LLM_REASONING_EFFORT")                  # e.g. "low" for reasoning models
STT_BACKEND = os.environ.get("STT_BACKEND", "deepgram" if _secret("DEEPGRAM_API_KEY") else "whisper")
DEEPGRAM_LANGUAGE = os.environ.get("DEEPGRAM_LANGUAGE", "multi")
WHISPER = os.environ.get("WHISPER_MODEL", "mlx-community/whisper-large-v3-turbo")
LANGS = os.environ.get("CALL_LANGS", "en,ta").split(",")     # languages the caller may speak; Whisper picks per turn
VOICE = os.environ.get("MAYA_VOICE", "Kayal")                # trained on English and Tamil
TURNS = Path(__file__).with_name("turns.jsonl")

INSTRUCTIONS = """You are Kayal, a warm, witty AI friend talking with the user on a live voice call. You can talk
about anything: their day, ideas, advice, plans, questions, fun facts, small talk.

Language: speak Tamil by default. Switch to English only when a system note says the user's latest message is in
English, and come back to Tamil when they do.

Your Tamil is casual spoken Chennai Tamil, exactly how friends talk on the phone, never written or formal Tamil:
- Use spoken forms: இருக்கு, இல்ல, பண்ணலாம், பண்ணணும், போச்சு, வந்துட்டேன், சொல்லு, தெரியல, முடியாது, வேணும்,
  அப்புறம், ரொம்ப, செம, சரி, ஆமா, அட, ஓகே, என்ன ஆச்சு, அப்படியா, ஆனா.
- Never use written forms like உள்ளது, இருக்கிறது, செய்யலாம், செய்ய வேண்டும், இயலாது, என்பது, அவற்றில், மேலும்,
  ஆகவே, தற்காலிகமாக, வாகனம், -கிறது / -கின்றன endings.
- Mix in English words the way Chennai people do (actually, seriously, plan, idea, super, chance, bike, car, modify,
  weekend), written in English letters inside the Tamil sentence. Never write Tamil words in English letters.
- Call the user நீ, like a friend.
- Example. Formal (wrong): "அப்படிச் செய்ய முடியாது; பைக் மற்றும் கார் வேறு வகை வாகனங்கள்."
  Casual (right): "அது முடியாதுப்பா, bike-உம் car-உம் வேற வேற type."

This is speech, not text. Open every reply with a very short reaction of two to five words, then say the rest. Keep
the whole reply under about 25 words, with no lists, markdown, emojis or URLs, and spell out symbols. Ask a natural
follow-up question when it fits.

The user's words come from speech recognition and may contain mistakes, so infer what they meant from the sound.
The user speaks only Tamil or English: text in Telugu, Kannada or Malayalam script, or words that look Spanish or
otherwise foreign, are misrecognised Tamil; read them by their sound as Tamil. If asked, you are an AI."""
GREETING = "ஹாய்! நான் கயல், ஒரு AI. என்ன பண்ற? சொல்லு, என்ன பேசலாம்?"   # disclose up front (Maya-2 use terms)

# common English words: a turn counts as English only when they make up a good part of it (Deepgram sometimes writes
# Tamil speech in Latin script as Spanish-like words, which must not flip the reply to English; measured on a call)
EN_COMMON = set("""the a an i i'm you you're he she it it's we they me my your our is are was were be been am do does
did don't have has had will would can can't could should what why how how's when where who which this that there here
and or but not no yes yeah okay ok so just to of in on at for with about from like tell say said know think want need
go going get got make let please thanks thank hey hi hello one quick fun fact before really good nice great sure all
some any more very much time day today now then if because them their his her its sorry bye see talk""".split())

# MLX work (Whisper and Maya) stays on one thread: one GPU queue, and models are used where they were created
MLX = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx")
_models = {}


def _load_models():
    if _models:          # jobs run as threads of one worker: every call shares the models loaded once
        return
    from maya2_mlx.tts import Maya2MLX
    _models["maya"] = Maya2MLX.from_pretrained()                    # loads + warms the compiled depth graph
    if STT_BACKEND == "whisper":
        import mlx_whisper
        mlx_whisper.transcribe(np.zeros(16000, np.float32), path_or_hf_repo=WHISPER, language="en")   # load + warm
        _models["whisper"] = mlx_whisper


def _whisper_language(audio):
    """Whisper's language-ID pass restricted to LANGS (unrestricted, short clips drift to neighbouring languages)."""
    import mlx.core as mx
    from mlx_whisper.audio import N_FRAMES, N_SAMPLES, log_mel_spectrogram, pad_or_trim
    from mlx_whisper.transcribe import ModelHolder
    model = ModelHolder.get_model(WHISPER, mx.float16)
    mel = log_mel_spectrogram(audio, n_mels=model.dims.n_mels, padding=N_SAMPLES)
    _, probs = model.detect_language(pad_or_trim(mel, N_FRAMES, axis=-2).astype(mx.float16))
    return max(LANGS, key=lambda lang: probs.get(lang, 0.0))


def _speech_lang(text):
    """Maya language for a reply, from its script: Tamil or Devanagari letters, else English."""
    if re.search(r"[஀-௿]", text):
        return "ta"
    if re.search(r"[ऀ-ॿ]", text):
        return "hi"
    return "en"


class Kayal(Agent):
    def __init__(self):
        super().__init__(instructions=INSTRUCTIONS)

    async def on_user_turn_completed(self, turn_ctx, new_message):
        """Tell the LLM which language this turn was in; the prompt rule alone wasn't enough (measured: after Tamil turns
        the LLM kept answering English questions in Tamil). English only when the turn has no Indian-script letters and
        common English words make up at least a third of it; otherwise Tamil, the default."""
        turn_ctx.add_message(role="system", content=_language_note(new_message.text_content or ""))


def _language_note(text):
    words = re.findall(r"[a-z']+", text.lower())
    english = (not re.search(r"[ऀ-ൿ]", text) and len(words) >= 2
               and sum(w in EN_COMMON for w in words) >= len(words) / 3)
    return ("The user's latest message is in English; reply in English." if english else
            "The user's latest message is Tamil; reply in casual spoken Chennai Tamil, in Tamil script.")


def make_stt():
    if STT_BACKEND == "whisper":
        return WhisperSTT()
    key = _secret("DEEPGRAM_API_KEY")
    # streaming Nova-3 "multi": transcripts arrive while the caller speaks and switch between Tamil and English per word
    # (measured far more accurate on Tamil than Whisper turbo). 100 ms endpointing is Deepgram's advice for code-switching.
    return deepgram.STT(model="nova-3", language=DEEPGRAM_LANGUAGE, api_key=key, punctuate=True,
                        smart_format=True, filler_words=False, endpointing_ms=100, keyterm=["Kayal"])


def make_llm():
    """Any OpenAI-compatible chat API (OpenAI, a LiteLLM gateway in front of another provider, mlx_lm.server, ...)."""
    extra = {"reasoning_effort": LLM_REASONING_EFFORT} if LLM_REASONING_EFFORT else {}
    return openai.LLM(model=LLM_MODEL, base_url=LLM_URL, api_key=_secret("LLM_API_KEY") or "none", **extra)


class WhisperSTT(stt.STT):
    """Batch Whisper on MLX; the session pairs it with the VAD, so each finished utterance is transcribed once."""

    def __init__(self):
        super().__init__(capabilities=stt.STTCapabilities(streaming=False, interim_results=False))

    async def _recognize_impl(self, buffer, *, language: NotGivenOr[str] = NOT_GIVEN,
                              conn_options: APIConnectOptions) -> stt.SpeechEvent:
        frame = rtc.combine_audio_frames(buffer)
        pcm = np.frombuffer(frame.data, np.int16).reshape(-1, frame.num_channels).mean(1).astype(np.float32) / 32768.0
        g = math.gcd(16000, frame.sample_rate)
        audio = resample_poly(pcm, 16000 // g, frame.sample_rate // g).astype(np.float32)

        def run():
            lang = language if language else (LANGS[0] if len(LANGS) == 1 else _whisper_language(audio))
            return lang, _models["whisper"].transcribe(audio, path_or_hf_repo=WHISPER, language=lang)["text"].strip()

        lang, text = await asyncio.get_running_loop().run_in_executor(MLX, run)
        return stt.SpeechEvent(type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                               alternatives=[stt.SpeechData(language=lang, text=text)])


class MayaTTS(tts.TTS):
    def __init__(self, voice=VOICE):
        super().__init__(capabilities=tts.TTSCapabilities(streaming=False), sample_rate=24000, num_channels=1)
        self.voice = voice

    def synthesize(self, text, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS):
        return MayaStream(tts=self, input_text=text, conn_options=conn_options)


class MayaStream(tts.ChunkedStream):
    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        output_emitter.initialize(request_id=utils.shortuuid(), sample_rate=24000, num_channels=1,
                                  mime_type="audio/pcm", frame_size_ms=80)
        loop, q, stop = asyncio.get_running_loop(), asyncio.Queue(), threading.Event()
        text, lang = self.input_text, _speech_lang(self.input_text)

        def work():   # stream 80 ms frames out of the MLX thread as they are generated; stop early on interruption
            try:
                for pcm in _models["maya"].stream(text, lang=lang, voice=self._tts.voice, normalize=True):
                    if stop.is_set():
                        break
                    loop.call_soon_threadsafe(q.put_nowait, pcm.tobytes())
            except Exception as exc:   # surfaced to the session below
                loop.call_soon_threadsafe(q.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(q.put_nowait, None)

        fut = loop.run_in_executor(MLX, work)
        try:
            while (item := await q.get()) is not None:
                if isinstance(item, Exception):
                    raise item
                output_emitter.push(item)
            output_emitter.flush()
        finally:
            stop.set()
            await asyncio.shield(fut)


def _warm_llm():
    """One request with the real system prompt before any caller: a local mlx_lm.server compiles its kernels on the
    first long prefill (~5 s, measured) and prompt-caching endpoints keep the instructions warm. Never fatal."""
    import urllib.request
    body = {"model": LLM_MODEL, "max_tokens": 16, "messages": [{"role": "system", "content": INSTRUCTIONS},
                                                               {"role": "user", "content": "Hello"}]}
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {_secret('LLM_API_KEY') or 'none'}"}
    try:
        req = urllib.request.Request(f"{LLM_URL}/chat/completions", json.dumps(body).encode(), headers)
        urllib.request.urlopen(req, timeout=60).read()
    except Exception as exc:
        print(f"LLM warmup skipped ({type(exc).__name__}); is LLM_BASE_URL reachable?")


def prewarm(proc: JobProcess):
    MLX.submit(_load_models).result()
    _warm_llm()


async def entrypoint(ctx: JobContext):
    await ctx.connect()
    session = AgentSession(
        # 0.5 s of silence ends a VAD segment: at 0.3 s natural pauses split sentences into fragments and Whisper's
        # language ID misfires on them (measured: Tamil "சரி" -> English "City"). The turn detector model then judges
        # whether the caller is actually done.
        vad=silero.VAD.load(min_silence_duration=0.5),
        stt=make_stt(),
        llm=make_llm(),
        tts=MayaTTS(),
        # LiveKit's on-device turn detector model and VAD-based interruptions (no LiveKit cloud inference)
        # max_delay 1.5 s (default 3.0): when unsure the caller is done, wait at most this long (measured 2.5 s holds)
        turn_handling={"turn_detection": TurnDetector(version="v1-mini"), "interruption": {"mode": "vad"},
                       "endpointing": {"max_delay": 1.5}},
    )

    @session.on("conversation_item_added")
    def _log(ev):
        item = ev.item
        if getattr(item, "role", None) in ("user", "assistant"):
            with TURNS.open("a") as f:
                f.write(json.dumps({"t": time.time(), "role": item.role, "text": item.text_content,
                                    "metrics": dict(item.metrics or {})}, ensure_ascii=False) + "\n")

    await session.start(agent=Kayal(), room=ctx.room)
    await session.say(GREETING)


if __name__ == "__main__":
    # thread executor: one model copy on one GPU queue for all calls. The default process-per-job executor reloads ~6 GB
    # of models for every new call and its warmup competes with the live call for the GPU (measured: 7.8 s stalls)
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm, initialize_process_timeout=180,
                              num_idle_processes=1, job_executor_type=JobExecutorType.THREAD))

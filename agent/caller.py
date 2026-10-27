"""Simulated phone caller: joins a LiveKit room over WebRTC, speaks pre-recorded lines, records what it hears.

Latency is measured where a real caller would feel it: from the end of the caller's speech (last voiced frame handed
to WebRTC) to the first audible frame of the agent's reply arriving back over WebRTC.

    python caller.py prepare [script]  # synthesise the caller's lines with Maya-2 into caller_lines/<script>/
    python caller.py call [script]     # run the call against the agent worker; writes call.wav and call_report.json
"""
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).parent
SCRIPTS = {   # (text, Maya language, caller voice)
    "store": [("Hi, what time does your store close today?", "en", "Akshat"),
              ("Do you deliver to Adyar? My order is around six hundred rupees.", "en", "Akshat"),
              ("Great. And can I pay with UPI?", "en", "Akshat"),
              ("Okay, thank you so much. Bye!", "en", "Akshat")],
    "chat": [("Hey, how's it going? I'm just sitting at home, kind of bored.", "en", "Abhay"),
             ("சரி, இந்த weekend Chennai-ல என்ன பண்ணலாம்னு ஏதாவது idea சொல்லு.", "ta", "Abhay"),
             ("Nice. Tell me one quick fun fact before I go.", "en", "Abhay"),
             ("சூப்பர்! சரி, நான் அப்புறம் பேசறேன். Bye!", "ta", "Abhay")],
}
SR, FRAME = 24000, 480                # 20 ms frames
# livekit-server --dev defaults
URL = os.environ.get("LIVEKIT_URL", "ws://127.0.0.1:7880")
KEY = os.environ.get("LIVEKIT_API_KEY", "devkey")
SECRET = os.environ.get("LIVEKIT_API_SECRET", "secret")
SPEAKING_RMS = 300                    # int16 RMS above which a received frame counts as agent speech
DONE_SILENCE_S = 1.5                  # agent turn is over after this much silence


def prepare(script):
    from maya2_mlx.tts import Maya2MLX
    tts = Maya2MLX.from_pretrained()
    (HERE / "caller_lines" / script).mkdir(parents=True, exist_ok=True)
    for i, (text, lang, voice) in enumerate(SCRIPTS[script]):
        sf.write(HERE / f"caller_lines/{script}/{i}.wav", tts.generate(text, lang, voice, normalize=True, seed=i), SR)
        print("wrote", i, text)


async def call(script, room_name):
    from livekit import api, rtc
    LINES = [text for text, _, _ in SCRIPTS[script]]
    lines = [sf.read(HERE / f"caller_lines/{script}/{i}.wav", dtype="int16")[0] for i in range(len(LINES))]
    room = rtc.Room()
    heard, sent = [], []                           # (perf_counter, int16 frame)
    t_speech = {"last": None}

    async def listen(track):
        async for ev in rtc.AudioStream(track, sample_rate=SR, num_channels=1):
            f = np.frombuffer(ev.frame.data, np.int16).copy()
            heard.append((time.perf_counter(), f))
            if np.sqrt(np.mean(f.astype(np.float32) ** 2)) > SPEAKING_RMS:
                t_speech["last"] = heard[-1][0]

    @room.on("track_subscribed")
    def _on_track(track, pub, participant):
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            asyncio.create_task(listen(track))

    token = (api.AccessToken(KEY, SECRET).with_identity("caller").with_name("Caller")
             .with_grants(api.VideoGrants(room_join=True, room=room_name)).to_jwt())
    await room.connect(URL, token)
    source = rtc.AudioSource(SR, 1, queue_size_ms=40)
    await room.local_participant.publish_track(rtc.LocalAudioTrack.create_audio_track("mic", source),
                                               rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE))
    pending = []                                    # speech samples waiting to be sent; silence otherwise
    speech_end = asyncio.Queue()

    async def mic():                                # a real-time microphone: always sending, speech or silence
        silence = np.zeros(FRAME, np.int16)
        while True:
            chunk = pending.pop(0) if pending else silence
            await source.capture_frame(rtc.AudioFrame(chunk.tobytes(), SR, 1, FRAME))
            sent.append((time.perf_counter(), chunk))
            if chunk is not silence and not pending:
                speech_end.put_nowait(time.perf_counter())

    async def agent_turn(after, timeout=30.0):
        """-> (first audible frame time after `after`, end time) of the agent's next turn."""
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < timeout:
            start = next((t for t, f in heard if t > after and np.sqrt(np.mean(f.astype(np.float32) ** 2)) > SPEAKING_RMS), None)
            if start is not None:
                while time.perf_counter() - t_speech["last"] < DONE_SILENCE_S:
                    await asyncio.sleep(0.05)
                return start, t_speech["last"]
            await asyncio.sleep(0.01)
        raise TimeoutError("agent did not answer")

    mic_task = asyncio.create_task(mic())
    t_call = time.perf_counter()
    report = {"room": room_name, "turns": []}
    g_start, g_end = await agent_turn(t_call, timeout=120)       # greeting (first job may still be loading)
    report["greeting_after_join_s"] = round(g_start - t_call, 3)
    for i, w in enumerate(lines):
        pending.extend(np.array_split(np.pad(w, (0, -len(w) % FRAME)), len(np.pad(w, (0, -len(w) % FRAME))) // FRAME))
        t_end = await speech_end.get()
        start, end = await agent_turn(t_end)
        report["turns"].append({"caller": LINES[i], "latency_s": round(start - t_end, 3), "reply_s": round(end - start, 2)})
        print(f"turn {i}: caller stopped -> agent audio in {start - t_end:.3f}s", flush=True)
    mic_task.cancel()
    await room.disconnect()

    # call recording: caller on the left channel, agent on the right, on one timeline
    t0 = min(sent[0][0], heard[0][0])
    n = int((max(sent[-1][0], heard[-1][0]) - t0 + 1) * SR)
    mix = np.zeros((n, 2), np.int16)
    for ch, frames in ((0, sent), (1, heard)):
        for t, f in frames:
            i = int((t - t0) * SR) - len(f)          # timestamps mark when a frame finished arriving / was handed off
            i = max(i, 0)
            mix[i:i + len(f), ch] = f[: n - i]
    sf.write(HERE / "call.wav", mix, SR)
    (HERE / "call_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    script = sys.argv[2] if len(sys.argv) > 2 else "store"
    if sys.argv[1] == "prepare":
        prepare(script)
    else:
        asyncio.run(call(script, f"call-{int(time.time())}"))

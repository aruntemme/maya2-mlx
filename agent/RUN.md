# Kayal: a real-time voice call agent (LiveKit + Deepgram + any OpenAI-compatible LLM + Maya-2)

Kayal is an AI friend you can call from a browser on your Wi-Fi. She speaks casual spoken Chennai Tamil by default
and switches to English when you speak English.

| Stage | What runs it | Where |
|---|---|---|
| Speech to text | Deepgram Nova-3 streaming, `language=multi` (Tamil/English code-switching); local Whisper large-v3-turbo (MLX) without a key | Cloud / Mac |
| Thinking | Any OpenAI-compatible chat API (`LLM_BASE_URL`, `LLM_MODEL`); by default a local MLX model on `mlx_lm.server` | Cloud / Mac |
| Voice | Maya-2 on MLX (this repo), voice Kayal | Mac |
| Calls | LiveKit server + LiveKit Agents | Mac |

Measured in scripted bilingual calls on an M4 Max, with GPT-6 Luna (reasoning effort low) through an OpenAI-compatible
gateway: median **1.8 s** from the caller stopping to hearing Kayal.

| Stage | Median |
|---|---|
| Transcript ready | 0.48 s |
| End-of-turn decision | 0.51 s |
| LLM's first token | 0.72 s |
| Maya's first audio | 0.14 s |

## Setup (macOS on Apple Silicon)

From this `agent/` folder:

```bash
brew install livekit caddy
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python .. "livekit-agents[silero,openai]~=1.8" "livekit-plugins-deepgram~=1.8" livekit-api mlx-whisper
cp ../.env.example ../.env
```

Then fill in `../.env`; the file is gitignored.

## Run: talk from any device on your Wi-Fi

Find the Mac's LAN IP with `ipconfig getifaddr en0`, then run each line in its own terminal from this folder,
using that IP:

```bash
livekit-server --dev --bind 0.0.0.0 --node-ip 192.168.x.y
```

```bash
.venv/bin/python agent.py dev
```

```bash
.venv/bin/python token_server.py
```

```bash
LAN_IP=192.168.x.y caddy run --config Caddyfile --adapter caddyfile
```

Open `https://192.168.x.y:8443` on any device on the same network and tap **Call**. Each visitor gets a private call,
and the page shows live captions and Kayal's state.

The certificate comes from Caddy's local authority, so each device warns once that the connection isn't private.
Browsers allow the microphone only on https pages, so the warning can't be avoided.
- Chrome or Brave: **Advanced → Proceed**.
- Safari: **Show Details → visit this website**.

**Terminal only, no server or browser:** `.venv/bin/python agent.py console`.

## Options (environment variables)

- `STT_BACKEND=whisper`: on-device speech recognition instead of Deepgram.
- `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`: any OpenAI-compatible chat API, such as OpenAI itself, a
  [LiteLLM](https://docs.litellm.ai) gateway in front of a provider that has no OpenAI-compatible endpoint, or Ollama.
- `LLM_REASONING_EFFORT`: e.g. `low` for reasoning models (halved the first-token time in our measurements).
- With nothing set, the agent uses a local `mlx_lm.server` on port 18090. Start it with
  `uvx --python 3.12 --from mlx-lm mlx_lm.server --model mlx-community/Qwen3-4B-Instruct-2507-4bit --port 18090`.
- `MAYA_VOICE`: any Maya-2 voice.
- `CALL_LANGS`: Whisper's languages, e.g. `en,ta,hi`.

## Measure it

```bash
.venv/bin/python caller.py prepare chat
```

```bash
.venv/bin/python caller.py call chat
```

The first command synthesises a scripted caller. The second places the call over WebRTC and writes `call.wav`
(caller left, agent right) and `call_report.json` (the latency the caller heard). The agent appends per-stage
timings to `turns.jsonl`. These outputs are gitignored because they contain call content.

## Design notes, each from a measured problem

- **One copy of the models (`JobExecutorType.THREAD`).** The default process-per-call executor reloads ~6 GB of
  models for each call, and its warmup stalled live calls by up to 7.8 s.
- **Local turn detection only.** LiveKit's `v1-mini` turn detector and VAD-based interruptions; without this, the
  framework tries LiveKit Cloud inference.
- **Openings first.** Kayal opens each reply with a few words, so the voice starts before the LLM's first long
  sentence is finished.
- **A per-turn language note to the LLM.** The prompt rule alone let the LLM answer English questions in Tamil.
- **Misheard scripts are read by sound.** Deepgram sometimes writes Tamil speech in Telugu script or as
  Spanish-looking words; Kayal is told to read those as Tamil.

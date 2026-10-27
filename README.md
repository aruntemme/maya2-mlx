# Maya-2 on MLX (Apple Silicon) + a real-time Tamil/English voice agent

An unofficial MLX port of the inference code for **[maya-research/maya2-native](https://huggingface.co/maya-research/maya2-native)**,
the open-weight Indic + English text-to-speech model by [Maya Research](https://mayaresearch.ai). All credit for the
model goes to them. This repo contains **code only**: it loads the official weights as they are, with no conversion
and no re-upload.

The official release needs Linux and an NVIDIA GPU (CUDA graphs, Triton). This port runs it natively on a Mac's GPU,
one stream at a time, with every component checked against the original PyTorch model.

**What's here:**
- **[`maya2_mlx/`](maya2_mlx):** the MLX inference port. Also on Hugging Face:
  [aruntemme/maya2-native-mlx](https://huggingface.co/aruntemme/maya2-native-mlx).
- **[`agent/`](agent):** Kayal, a voice call agent built on it with LiveKit, Deepgram and GPT-6 Luna. She speaks
  casual Tamil and English, and you can call her from any browser on your Wi-Fi. See [`agent/RUN.md`](agent/RUN.md).
- **[`parity/`](parity):** the checks against the original PyTorch code, with [raw results](parity/results.txt).
- **[`patches/maya2-mps.patch`](patches/maya2-mps.patch):** a small patch that lets the official PyTorch engine run
  on Apple's MPS backend.
- **[`bench/`](bench):** speed and intelligibility benchmarks.
- **[`samples/`](samples):** audio generated with this port.

## Install

Needs macOS on Apple Silicon and Python 3.10–3.12.

```bash
pip install "maya2-mlx @ git+https://github.com/aruntemme/maya2-mlx"
```

This also installs the official `maya3` package from GitHub, pinned to the commit the parity checks ran against.
Its text normaliser, prompt packer, voices and guards are reused. The weights (~5 GB), the Mimi codec and the ECAPA
speaker encoder download from Hugging Face on first use.

## Use

```python
import soundfile as sf
from maya2_mlx.tts import Maya2MLX

tts = Maya2MLX.from_pretrained()          # exact port: bf16 backbone, fp16 depth decoder
wav = tts.generate("வணக்கம், எப்படி இருக்கீங்க?", lang="ta", voice="Kayal", normalize=True)
sf.write("speech.wav", wav, 24000)

# stream 80 ms int16 PCM frames as they are generated
for chunk in tts.stream("Your order has shipped.", lang="en", voice="Siddhant", normalize=True):
    ...

# clone a voice across languages (get the speaker's permission first)
tts.register_voice("my_voice", ref_audio="reference.wav", ref_text="<exact transcript>", lang="en")
wav = tts.generate("नमस्ते, आप कैसे हैं?", lang="hi", voice="my_voice", normalize=True)
```

- **Same API as the official interface:** `generate()` and `stream()` take the same keyword arguments as
  `maya3.Maya3TTS` (`seed`, `temperature`, `top_k`, `top_p`, `repetition_penalty`, `max_frames`, `ref_audio` /
  `ref_text` / `ref_lang`, `return_info`).
- **Voices:** all 20 named voices from the release are available.
- **Fast mode:** `Maya2MLX.from_pretrained(backbone_bits=8, depth_dtype="bf16")`, an opt-in trade-off measured below.

## Speed (M4 Max, 36 GB)

| | Time to make 1 s of audio | Time to first audio | Peak memory |
|---|---|---|---|
| Official PyTorch engine on MPS (with the patch) | 0.94 s | 120 ms | – |
| **This port, exact** (default) | **0.41 s** | 60 ms | 6.1 GB |
| This port, fast (8-bit backbone + bf16 depth) | **0.30 s** | 50 ms | 4.5 GB |

## Intelligibility

Character error rate when Whisper large-v3-turbo transcribes the output (one sample per clip; lower is better):

| | en | hi | ta | te | bn | kn | mr | clone en→hi |
|---|---|---|---|---|---|---|---|---|
| Official PyTorch engine | 0.00 | 0.15 | 0.00 | 0.13 | 0.20 | 0.13 | 0.27 | 0.07 |
| This port, exact | 0.00 | 0.03 | 0.00 | 0.07 | 0.17 | 0.09 | 0.14 | 0.07 |
| This port, fast | 0.00 | 0.06 | 0.04 | 0.07 | 0.22 | 0.20 | 0.11 | 0.07 |

With one sample per language, read these as "the same quality".

## Parity with the PyTorch model

| Component | Result |
|---|---|
| Prompt embeddings | 97–99% bit-identical; the rest differ by 1 bf16 ulp |
| Backbone (Qwen3-1.7B) | error at or below PyTorch's own bf16 noise; same cb0 argmax; cb0 KL ≤ 2e-3 |
| Depth decoder, 31 levels | logit error vs fp32 1.4e-3 (PyTorch fp16 itself: 1.3e-3); greedy codes 95–97% equal |
| Mimi streaming decoder | 122 dB SNR vs the engine's decoder |
| Fast-mode costs | 8-bit backbone: cb0 KL 4.5e-3; bf16 depth: 8× the fp16 logit error; 4-bit rejected (KL 0.32) |

To rerun the checks, apply `patches/maya2-mps.patch` to a checkout of
[MayaResearch/maya2-native](https://github.com/MayaResearch/maya2-native) at the pinned commit and install it.
Then run `PYTHONPATH=parity:. python parity/check_backbone.py` (also `check_depth.py`, `check_mimi.py` and
`check_variants.py`).

## Limits

- **One stream at a time.** The CUDA engine's continuous batching isn't ported.
- **Seeds aren't portable.** A seed reproduces audio within this port, but not the CUDA engine's audio.
- **Cloning uses PyTorch for the reference.** The clip is encoded with the PyTorch Mimi encoder and ECAPA on CPU,
  once per voice.

## License and credit

Apache-2.0, like the original. Derived from [MayaResearch/maya2-native](https://github.com/MayaResearch/maya2-native);
the model weights, voices and evaluation are Maya Research's. This port is not affiliated with or endorsed by Maya
Research. The vendored [`agent/web/livekit-client.umd.js`](https://github.com/livekit/client-sdk-js) is LiveKit's
Apache-2.0 browser SDK.

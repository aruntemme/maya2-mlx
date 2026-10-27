"""PyTorch reference objects for the parity checks (the original model on MPS, bf16 as the engine loads it).

The engine's FastDepth / StaticMimi are CUDA-graph code; on a Mac they need patches/maya2-mps.patch applied to the
official maya2-native checkout (see the README)."""
import numpy as np
import torch

from maya3.engine import ROOT   # also puts the release's root and scripts/ on sys.path
from maya2_mlx import weights as W

DEV = "mps"


def paths():
    return W.resolve()


def torch_model(r):
    from tts.model.load_s3 import load_model_for_eval
    mcfg = {"model": {"backbone": r["backbone_dir"], "attn": "sdpa"}, "data": {"max_tokens": 8192}}
    model, _ = load_model_for_eval(r["ckpt"], device=DEV, dtype=torch.bfloat16, cfg=mcfg,
                                   config=str(ROOT / "configs/s3_fresh_v1.yaml"), amortize_frac=1.0,
                                   require_anchor=True, verbose=False)
    return model.eval()


def voices(r, packer):
    """Named voices + plain, with the host-side prefix / speaker refs prepare_voice() computes."""
    from maya3.prompt import VoiceRegistry, load_named_voices, MODES
    reg = VoiceRegistry()
    load_named_voices(reg, r["voices_json"])
    for vid in reg.ids():
        prepare_host(reg.get(vid), packer)
    return reg


def prepare_host(v, packer):
    from maya3.prompt import DEFAULT_BUILDER
    v.prefix = (v.builder or DEFAULT_BUILDER).prefix(packer, v)
    v.spk_refs = np.asarray(v.ecapa, np.float32).reshape(1, 192) if v.ecapa is not None else np.zeros((1, 192), np.float32)
    v.ts = int(v.prefix["tokens"].shape[0]) if v.prefix is not None else 0
    return v


def rel(a, b):
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))


def cos(a, b):
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

"""Locate a Maya-2 release and load its tensors as MLX arrays."""
import json
from pathlib import Path

import mlx.core as mx


def resolve(repo="maya-research/maya2-native"):
    from maya3 import hub
    return hub.resolve(repo)


def load(ckpt_dir):
    """All tensors of the release's safetensors shards, keyed as in the checkpoint (bf16)."""
    d = Path(ckpt_dir)
    index = d / "model.safetensors.index.json"
    files = sorted(set(json.loads(index.read_text())["weight_map"].values())) if index.exists() else ["model.safetensors"]
    w = {}
    for f in files:
        w.update(mx.load(str(d / f)))
    return w


def tokenizer(backbone_dir):
    """The backbone tokenizer extended with the TTS control tokens (as tts.model.backbone.Backbone builds it)."""
    from transformers import AutoTokenizer
    from tts.tokens import extend_tokenizer
    return extend_tokenizer(AutoTokenizer.from_pretrained(str(backbone_dir)))

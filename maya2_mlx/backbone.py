"""Qwen3 backbone (decoder stack only) on MLX with a preallocated KV cache for one stream.

Mirrors serve/fast_backbone.StaticBackbone: q/k RMSNorm, non-interleaved RoPE, GQA, SiLU-gated MLP, final norm.
Inputs are embeddings (the TTS prompt is composed outside), outputs are final-norm hidden states.
"""
import json
from pathlib import Path

import mlx.core as mx


class Backbone:
    def __init__(self, w, backbone_dir, capacity=2048, dtype=mx.bfloat16, bits=None):
        """bits: None = bf16 weights (the reference numerics); 8 or 4 = MLX affine-quantized projection weights
        (group size 64), an opt-in speed/quality trade-off; embeddings and the cb0 head stay bf16."""
        cfg = json.loads((Path(backbone_dir) / "config.json").read_text())
        self.n_layers = cfg["num_hidden_layers"]
        self.heads, self.kv_heads = cfg["num_attention_heads"], cfg["num_key_value_heads"]
        self.head_dim = cfg.get("head_dim") or cfg["hidden_size"] // self.heads
        self.hidden = cfg["hidden_size"]
        self.eps = cfg["rms_norm_eps"]
        self.theta = float(cfg.get("rope_parameters", {}).get("rope_theta") or cfg["rope_theta"])
        self.scale = self.head_dim ** -0.5
        self.capacity = capacity
        self.dtype = dtype
        p = "backbone.lm.model."
        self.embed = w[p + "embed_tokens.weight"]
        self.norm = w[p + "norm.weight"]
        self.cb0_head = w["backbone.cb0_head.weight"]
        self.layers = []
        for i in range(self.n_layers):
            q = f"{p}layers.{i}."
            self.layers.append({k: w[q + n] for k, n in (
                ("ln1", "input_layernorm.weight"), ("ln2", "post_attention_layernorm.weight"),
                ("wq", "self_attn.q_proj.weight"), ("wk", "self_attn.k_proj.weight"), ("wv", "self_attn.v_proj.weight"),
                ("wo", "self_attn.o_proj.weight"), ("qn", "self_attn.q_norm.weight"), ("kn", "self_attn.k_norm.weight"),
                ("wg", "mlp.gate_proj.weight"), ("wu", "mlp.up_proj.weight"), ("wd", "mlp.down_proj.weight"))})
        self.bits = bits
        if bits:
            for L in self.layers:
                for k in ("wq", "wk", "wv", "wo", "wg", "wu", "wd"):
                    L[k] = mx.quantize(L[k], 64, bits)
        self.reset()

    def _mm(self, y, wt):
        if self.bits:
            return mx.quantized_matmul(y, *wt, transpose=True, group_size=64, bits=self.bits)
        return y @ wt.T

    def reset(self):
        shape = (1, self.kv_heads, self.capacity, self.head_dim)
        self.k = [mx.zeros(shape, self.dtype) for _ in range(self.n_layers)]
        self.v = [mx.zeros(shape, self.dtype) for _ in range(self.n_layers)]
        self.length = 0

    def load_prefix(self, kv, length):
        """Install a cached prefix KV: list of per-layer (k, v) [1, KV, length, Dh]."""
        for i, (k, v) in enumerate(kv):
            self.k[i][:, :, :length] = k
            self.v[i][:, :, :length] = v
        self.length = length

    def prefix_kv(self, length):
        return [(self.k[i][:, :, :length], self.v[i][:, :, :length]) for i in range(self.n_layers)]

    def forward(self, x):
        """x [T, hidden] embeddings at positions length..length+T-1 -> final-norm hidden [T, hidden]; extends the cache."""
        T = x.shape[0]
        pos, end = self.length, self.length + T
        if end > self.capacity:
            raise ValueError(f"context {end} exceeds capacity {self.capacity}")
        H, KV, D = self.heads, self.kv_heads, self.head_dim
        h = x[None].astype(self.dtype)
        for i, L in enumerate(self.layers):
            y = mx.fast.rms_norm(h, L["ln1"], self.eps)
            q = mx.fast.rms_norm(self._mm(y, L["wq"]).reshape(1, T, H, D), L["qn"], self.eps).transpose(0, 2, 1, 3)
            k = mx.fast.rms_norm(self._mm(y, L["wk"]).reshape(1, T, KV, D), L["kn"], self.eps).transpose(0, 2, 1, 3)
            v = self._mm(y, L["wv"]).reshape(1, T, KV, D).transpose(0, 2, 1, 3)
            q = mx.fast.rope(q, D, traditional=False, base=self.theta, scale=1.0, offset=pos)
            k = mx.fast.rope(k, D, traditional=False, base=self.theta, scale=1.0, offset=pos)
            self.k[i][:, :, pos:end] = k
            self.v[i][:, :, pos:end] = v
            o = mx.fast.scaled_dot_product_attention(q, self.k[i][:, :, :end], self.v[i][:, :, :end], scale=self.scale,
                                                     mask="causal" if T > 1 else None)
            h = h + self._mm(o.transpose(0, 2, 1, 3).reshape(1, T, H * D), L["wo"])
            y = mx.fast.rms_norm(h, L["ln2"], self.eps)
            g = self._mm(y, L["wg"])
            h = h + self._mm(g * mx.sigmoid(g) * self._mm(y, L["wu"]), L["wd"])
        self.length = end
        return mx.fast.rms_norm(h, self.norm, self.eps)[0]

    def cb0_logits(self, h):
        return h @ self.cb0_head.T

"""Streaming one-frame Mimi decoder on MLX, mirroring serve/fast_codec.StaticMimi (the engine's verified decoder) for one
stream: causal-conv history, transposed-conv overlap tails (bias removed from the stored tail), and a sliding-window
transformer with absolute RoPE positions. The layer graph is read from the HF MimiModel built on the meta device, so it
is exactly the module tree the reference walks; weights come from kyutai/mimi's model.safetensors. Layout is NLC.
"""
import glob
import math
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn


def _conv_w(t):            # torch Conv1d [Cout, Cin/g, K] -> MLX [Cout, K, Cin/g]
    return t.transpose(0, 2, 1)


def _convT_w(t, groups):   # torch ConvTranspose1d [Cin, Cout/g, K] -> MLX [Cout, K, Cin/g]
    cin, cout_g, k = t.shape
    return t.reshape(groups, cin // groups, cout_g, k).transpose(0, 2, 3, 1).reshape(groups * cout_g, k, cin // groups)


class Mimi:
    def __init__(self, mimi_dir, dtype=mx.float32):
        import torch
        from transformers import MimiConfig, MimiModel
        from transformers.models.mimi.modeling_mimi import MimiConv1d, MimiConvTranspose1d, MimiResnetBlock
        self._types = (MimiConv1d, MimiConvTranspose1d, MimiResnetBlock)
        cfg = MimiConfig.from_pretrained(mimi_dir)
        with torch.device("meta"):
            m = MimiModel(cfg)
        names = {id(mod): n for n, mod in m.named_modules()}
        f = glob.glob(str(Path(mimi_dir) / "*.safetensors"))[0]
        w = {k: v.astype(dtype) for k, v in mx.load(f).items() if not k.startswith(("encoder", "downsample"))}
        self.dtype = dtype
        self.window = cfg.sliding_window
        self.eps = cfg.norm_eps
        # quantizer: codebook vectors = embed_sum / clamp(cluster_usage, 1e-5) (MimiEuclideanCodebook.embed)
        self.rvq = []
        for part, n in (("semantic", cfg.num_semantic_quantizers), ("acoustic", cfg.num_quantizers - cfg.num_semantic_quantizers)):
            p = f"quantizer.{part}_residual_vector_quantizer."
            books = [w[f"{p}layers.{i}.codebook.embed_sum"] / mx.maximum(w[f"{p}layers.{i}.codebook.cluster_usage"], 1e-5)[:, None]
                     for i in range(n)]
            self.rvq.append((n, mx.stack(books), w[p + "output_proj.weight"][:, :, 0]))
        # conv stack: upsample + decoder layers, as a plan of ops with per-layer state
        self.state0 = {}
        self.plan = [self._op(m.upsample, w, names)] + [self._op(l, w, names) for l in m.decoder.layers]
        # decoder transformer
        a = m.decoder_transformer.layers[0].self_attn
        self.heads, self.kv_heads, self.head_dim = a.num_heads, a.num_key_value_heads, a.head_dim
        self.theta = float(cfg.rope_parameters["rope_theta"]) if hasattr(cfg, "rope_parameters") and cfg.rope_parameters \
            else float(cfg.rope_theta)
        self.tf = []
        for i in range(cfg.num_hidden_layers):
            p = f"decoder_transformer.layers.{i}."
            self.tf.append({k: w[p + n] for k, n in (
                ("ln1_w", "input_layernorm.weight"), ("ln1_b", "input_layernorm.bias"),
                ("ln2_w", "post_attention_layernorm.weight"), ("ln2_b", "post_attention_layernorm.bias"),
                ("wq", "self_attn.q_proj.weight"), ("wk", "self_attn.k_proj.weight"), ("wv", "self_attn.v_proj.weight"),
                ("wo", "self_attn.o_proj.weight"), ("s1", "self_attn_layer_scale.scale"), ("s2", "mlp_layer_scale.scale"),
                ("fc1", "mlp.fc1.weight"), ("fc2", "mlp.fc2.weight"))})
        self.reset()

    def _op(self, mod, w, names):
        MimiConv1d, MimiConvTranspose1d, MimiResnetBlock = self._types
        key = names[id(mod)]
        if isinstance(mod, MimiConv1d):
            c = mod.conv
            if c.stride != (1,):
                raise ValueError("only stride-one decoder convolutions are supported")
            pad = (c.kernel_size[0] - 1) * c.dilation[0]
            if pad:
                self.state0[key] = mx.zeros((1, pad, c.in_channels), self.dtype)
            return ("conv", key, _conv_w(w[key + ".conv.weight"]), w.get(key + ".conv.bias"), c.dilation[0], c.groups, pad)
        if isinstance(mod, MimiConvTranspose1d):
            c = mod.conv
            if mod.padding_left != 0 or c.dilation != (1,):
                raise ValueError("unsupported transpose-convolution geometry")
            if mod.padding_right:
                self.state0[key] = mx.zeros((1, mod.padding_right, c.out_channels), self.dtype)
            return ("convT", key, _convT_w(w[key + ".conv.weight"], c.groups), w.get(key + ".conv.bias"), c.stride[0], c.groups,
                    mod.padding_right)
        if isinstance(mod, MimiResnetBlock):
            return ("res", key, self._op(mod.shortcut, w, names) if id(mod.shortcut) in names and
                    isinstance(mod.shortcut, self._types) else None, [self._op(ch, w, names) for ch in mod.block])
        if type(mod).__name__ == "ELU":
            return ("elu",)
        if type(mod).__name__ == "Identity":
            return ("id",)
        raise ValueError(f"unexpected Mimi decoder module {type(mod).__name__} at {key}")

    def reset(self):
        """Start a new stream: zero conv history / overlap tails and an empty attention window."""
        self.state = dict(self.state0)
        self.k = [None] * len(self.tf)
        self.v = [None] * len(self.tf)
        self.pos = 0

    def _run(self, op, x):
        kind = op[0]
        if kind == "conv":
            _, key, wt, b, dil, groups, pad = op
            if pad:
                x = mx.concatenate([self.state[key], x], 1)
                self.state[key] = x[:, -pad:]
            y = mx.conv1d(x, wt, stride=1, padding=0, dilation=dil, groups=groups)
            return y + b if b is not None else y
        if kind == "convT":
            _, key, wt, b, stride, groups, overlap = op
            y = mx.conv_transpose1d(x, wt, stride=stride, groups=groups)
            if b is not None:
                y = y + b
            produced = x.shape[1] * stride
            if overlap:
                tail = y[:, produced:]
                if b is not None:
                    tail = tail - b
                y = mx.concatenate([y[:, :overlap] + self.state[key], y[:, overlap:produced]], 1)
                self.state[key] = tail
            return y[:, :produced]
        if kind == "res":
            _, key, shortcut, block = op
            s = self._run(shortcut, x) if shortcut is not None else x
            for child in block:
                x = self._run(child, x)
            return s + x
        if kind == "elu":
            return nn.elu(x)
        return x

    def _transformer(self, h):
        B, T, _ = h.shape
        H, KV, Dh = self.heads, self.kv_heads, self.head_dim
        p0 = self.pos
        # a query at position p sees keys with p - window < key_pos <= p
        qpos = mx.arange(p0, p0 + T)[:, None]
        for i, L in enumerate(self.tf):
            x = mx.fast.layer_norm(h, L["ln1_w"], L["ln1_b"], self.eps)
            q = (x @ L["wq"].T).reshape(B, T, H, Dh).transpose(0, 2, 1, 3)
            k = (x @ L["wk"].T).reshape(B, T, KV, Dh).transpose(0, 2, 1, 3)
            v = (x @ L["wv"].T).reshape(B, T, KV, Dh).transpose(0, 2, 1, 3)
            q = mx.fast.rope(q, Dh, traditional=False, base=self.theta, scale=1.0, offset=p0)
            k = mx.fast.rope(k, Dh, traditional=False, base=self.theta, scale=1.0, offset=p0)
            if self.k[i] is not None:
                k, v = mx.concatenate([self.k[i], k], 2), mx.concatenate([self.v[i], v], 2)
            kpos = mx.arange(p0 + T - k.shape[2], p0 + T)[None]
            mask = (kpos <= qpos) & (kpos > qpos - self.window)
            a = mx.fast.scaled_dot_product_attention(q, k, v, scale=Dh ** -0.5, mask=mask)
            h = h + L["s1"] * (a.transpose(0, 2, 1, 3).reshape(B, T, H * Dh) @ L["wo"].T)
            x = mx.fast.layer_norm(h, L["ln2_w"], L["ln2_b"], self.eps)
            h = h + L["s2"] * (nn.gelu(x @ L["fc1"].T) @ L["fc2"].T)
            keep = self.window - 1             # the oldest key any future query can still see
            self.k[i], self.v[i] = k[:, :, -keep:], v[:, :, -keep:]
        self.pos = p0 + T
        return h

    def decode(self, codes):
        """codes [32] int (one frame) -> float waveform [samples_per_frame] (1920 at 24 kHz)."""
        x = None
        start = 0
        for n, books, proj in self.rvq:
            e = books[mx.arange(n), codes[start:start + n]].sum(0)          # [codebook_dim]
            y = e @ proj.T
            x = y if x is None else x + y
            start += n
        x = x[None, None].astype(self.dtype)                                # [1, 1, 512] (NLC)
        x = self._run(self.plan[0], x)                                      # upsample -> 2 tokens
        x = self._transformer(x)
        for op in self.plan[1:]:
            x = self._run(op, x)
        return x[0, :, 0]

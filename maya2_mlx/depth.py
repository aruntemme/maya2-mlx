"""Depth decoder (31 levels per frame) on MLX: serve/fast_depth.FastDepth._compute, fp16 by default.

Same function as the reference: prefix position k is embedded from slot k alone, levels sharing a trunk group reuse the
group's K/V (exact because attention is causal), the last layer of a group's first pass evaluates only the final
position, and the speaker cross-attention is a per-voice constant (one reference token => softmax is exactly one).
"""
import math

import mlx.core as mx
import mlx.nn as nn

from .prompt import RMS_EPS

D, H, FFN, LAYERS, N_LEVELS = 1024, 16, 3072, 4, 32
HEAD = D // H
GROUP_OF_LEVEL = [None] + list(range(8)) + [8] * 8 + [9] * 8 + [10] * 7
EPS = 1e-5


class Depth:
    def __init__(self, w, dtype=mx.float16, temp=0.7, top_k=64, n_codes=2048):
        c = lambda k: w[k].astype(dtype)
        self.dtype, self.temp, self.top_k, self.n_codes = dtype, float(temp), int(top_k), n_codes
        self.in_proj = c("depth.in_proj.weight")
        level = c("depth.level_emb.weight")
        self.level0 = level[0]
        # position k+1 of the prefix = code_up(code_lo[cb_k + k*n_codes]) + level_emb[k+1]  (computed once, as FastDepth)
        lo = w["depth.code_lo.weight"].astype(mx.float32).reshape(N_LEVELS - 1, n_codes, -1)
        table = (lo @ w["depth.code_up.weight"].astype(mx.float32).T).astype(dtype)
        # flat [31*n_codes, D]: one-row gathers (indexing a [31, n, D] table per level copied the whole 4 MB slice)
        self.embed = (table + level[1:N_LEVELS, None]).reshape(-1, D)
        self.groups = [[{n: c(f"depth.groups.{g}.{l}.{n}") for n in
                         ("n1.weight", "n1.bias", "n2.weight", "n2.bias", "qkv.weight", "o.weight", "f1.weight", "f2.weight")}
                        for l in range(LAYERS)] for g in range(11)]
        self.cross = [{n: c(f"depth.cross.{l}.{n}") for n in ("kv.weight", "o.weight")} for l in range(LAYERS)]
        self.norm_w, self.norm_b = c("depth.norm.weight"), c("depth.norm.bias")
        self.heads = [c(f"depth.heads.{i}.weight") for i in range(N_LEVELS - 1)]
        self.spk_proj = w["depth.spk_proj_ecapa.weight"]
        self.null_mem = c("depth.null_mem")
        # backbone feedback embedding (AudioInputEmbed.frames over all 32 levels, backbone dtype)
        self.fb_table, self.fb_norm = w["audio_in.table.weight"], w["audio_in.norm.weight"]
        self._frame = mx.compile(self._compute)

    def speaker(self, refs=None):
        """Per-layer cross-attention constants for a voice: refs (192,) ECAPA or None (nulled memory)."""
        # the engine computes the memory with the fp32 reference depth, then casts it to the serving dtype
        mem = self.null_mem[0] if refs is None else \
            (mx.array(refs, mx.float32)[None] @ self.spk_proj.astype(mx.float32).T).astype(self.dtype)
        return mx.stack([(mem @ L["kv.weight"].T)[:, D:] @ L["o.weight"].T for L in self.cross])   # [LAYERS, 1, D]

    def feedback(self, codes):
        lvl = mx.arange(N_LEVELS) * self.n_codes
        e = self.fb_table[codes + lvl].sum(1) / math.sqrt(N_LEVELS)
        return mx.fast.rms_norm(e, self.fb_norm, RMS_EPS)

    def _trunk_layer(self, L, x, kv_prev, causal, last_only):
        B, T, _ = x.shape
        qkv = (mx.fast.layer_norm(x, L["n1.weight"], L["n1.bias"], EPS) @ L["qkv.weight"].T).reshape(B, T, 3, H, HEAD)
        q, k, v = (qkv[:, :, i].transpose(0, 2, 1, 3) for i in range(3))
        if kv_prev is not None:
            k, v = mx.concatenate([kv_prev[0], k], 2), mx.concatenate([kv_prev[1], v], 2)
        if last_only:
            q, x = q[:, :, -1:], x[:, -1:]
        a = mx.fast.scaled_dot_product_attention(q, k, v, scale=HEAD ** -0.5, mask="causal" if causal else None)
        x = x + a.transpose(0, 2, 1, 3).reshape(B, x.shape[1], D) @ L["o.weight"].T
        x = x + nn.gelu(mx.fast.layer_norm(x, L["n2.weight"], L["n2.bias"], EPS) @ L["f1.weight"].T) @ L["f2.weight"].T
        return x, (k, v)

    def _compute(self, h, cb0, cross, key, force=None):
        """h [B, 2048] backbone hidden, cb0 [B] int, cross [LAYERS, 1, D] -> (codes [B, 32], feedback [B, 2048], logits list).
        force [B, 32]: teacher forcing (prefix built from these codes instead of the sampled ones; for parity checks)."""
        B = h.shape[0]
        prefix = [(h.astype(self.dtype) @ self.in_proj.T + self.level0)[:, None], self.embed[cb0][:, None]]
        codes, logits_all, kv, previous = [cb0], [], None, None
        for level in range(1, N_LEVELS):
            group, n = GROUP_OF_LEVEL[level], level + 1
            incremental = group == previous
            x = prefix[level] if incremental else mx.concatenate(prefix[:n], 1)
            new_kv = []
            for layer, L in enumerate(self.groups[group]):
                last = (not incremental) and layer == LAYERS - 1
                x, kvl = self._trunk_layer(L, x, kv[layer] if incremental else None, causal=not incremental and not last,
                                           last_only=last)
                new_kv.append(kvl)
                x = x + cross[layer]
            kv, previous = new_kv, group
            y = mx.fast.layer_norm(x[:, -1], self.norm_w, self.norm_b, EPS)
            logits = (y @ self.heads[level - 1].T).astype(mx.float32) / self.temp
            logits_all.append(logits)
            if force is not None:
                code = force[:, level]
            elif key is None:
                code = mx.argmax(logits, -1)
            else:
                kth = mx.topk(logits, self.top_k, -1).min(-1, keepdims=True)
                probs = mx.softmax(mx.where(logits < kth, -mx.inf, logits), -1)
                key, sub = mx.random.split(key)
                race = -mx.log(mx.random.uniform(shape=probs.shape, key=sub))     # Exp(1): the same exponential race
                code = mx.argmax(probs / race, -1)
            codes.append(code)
            if n < N_LEVELS:
                prefix.append(self.embed[code + level * self.n_codes][:, None])
        codes = mx.stack(codes, -1)
        return codes, self.feedback(codes), logits_all

    def __call__(self, h, cb0, cross, key):
        codes, fb, _ = self._frame(h, cb0, cross, key)
        return codes, fb

"""Patch-token pooling strategies for BKL vision features.

Input contract for every pool: a tensor of shape (B, T, N, C) where
  B = batch, T = sequence length (num_steps), N = patch tokens (197 for
  ViT-B/16 at 224×224: 1 CLS + 196 patches), C = embedding dim (768).
Output: (B, T, C).

This contract lets any pool be dropped into the same code path. The top-level
wrapper PerCamPool runs one pool per camera (independent weights, matching
the existing per-cam pattern used by ActorTransformerConcatAttnPooling).
"""

import torch
import torch.nn as nn


class MeanPool(nn.Module):
    """Drop CLS, mean over 196 patch tokens. No learnable parameters.

    Identical to the previous bc.py helper `mean_pool_patches`, and also
    identical to what mode='mean' extraction used to compute inside the ViT.
    Selecting this pool keeps behavior bit-exact to the pre-pluggable pipeline.
    """

    def forward(self, x):
        return x[..., 1:, :].mean(dim=-2)


class MaxPool(nn.Module):
    """Drop CLS, max over 196 patch tokens. No learnable parameters."""

    def forward(self, x):
        return x[..., 1:, :].max(dim=-2).values


class MinPool(nn.Module):
    """Drop CLS, min over 196 patch tokens. No learnable parameters."""

    def forward(self, x):
        return x[..., 1:, :].min(dim=-2).values


class ClsPool(nn.Module):
    """Use only the CLS token (position 0). No learnable parameters."""

    def forward(self, x):
        return x[..., 0, :]


class AttnPool(nn.Module):
    """Single learned query cross-attending over all 197 tokens.

    Perceiver-style attention pooling: a single learned query token attends
    over (CLS + 196 patches) via one multi-head attention layer, followed by
    a residual feed-forward MLP. LayerNorms around both sub-blocks for
    training stability.

    Parameter count at C=768, num_heads=8, mlp_ratio=2.0: ~2.5M per pool
    (roughly: 4*C^2 for MHA proj + 2*C*C*mlp_ratio for FFN + 4*C for norms +
    C for the query). Three cams in total → ~7.5M extra params.

    CLS is left in the key/value set — it's a trained summarization token
    and carries useful signal; the query can learn to attend to it or not.
    """

    def __init__(self, embed_dim, num_heads=8, mlp_ratio=2.0):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        self.norm_ff = nn.LayerNorm(embed_dim)
        hidden = int(embed_dim * mlp_ratio)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, embed_dim),
        )

    def forward(self, x):
        # x: (B, T, N, C)
        B, T, N, C = x.shape
        x_flat = x.reshape(B * T, N, C)

        q = self.norm_q(self.query.expand(B * T, -1, -1))    # (B*T, 1, C)
        kv = self.norm_kv(x_flat)                             # (B*T, N, C)
        pooled, _ = self.attn(q, kv, kv, need_weights=False)  # (B*T, 1, C)
        pooled = pooled.squeeze(1)                            # (B*T, C)

        pooled = pooled + self.ff(self.norm_ff(pooled))       # residual MLP

        return pooled.reshape(B, T, C)


def build_pool(pool_type, embed_dim):
    """Factory: construct a pool module by name.

    pool_type options: 'mean', 'max', 'min', 'cls', 'attn'.
    embed_dim is only used by 'attn'; the others ignore it.
    """
    if pool_type == 'mean':
        return MeanPool()
    if pool_type == 'max':
        return MaxPool()
    if pool_type == 'min':
        return MinPool()
    if pool_type == 'cls':
        return ClsPool()
    if pool_type == 'attn':
        return AttnPool(embed_dim)
    raise ValueError(f"unknown pool_type: {pool_type!r} (valid: mean, max, min, cls, attn)")


class PerCamPool(nn.Module):
    """Runs one pool per camera with independent weights.

    Accepts a LIST of (B, T, N, C) tensors (one per camera) and returns a
    list of (B, T, C). The list order matches the order of cam_ids passed to
    the parent actor.
    """

    def __init__(self, pool_type, num_cams, embed_dim):
        super().__init__()
        self.pools = nn.ModuleList([
            build_pool(pool_type, embed_dim) for _ in range(num_cams)
        ])

    def forward(self, per_cam_features):
        return [self.pools[i](x) for i, x in enumerate(per_cam_features)]

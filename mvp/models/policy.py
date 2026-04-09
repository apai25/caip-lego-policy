from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.modules import MultiheadAttention

from mvp.models.helper import get_activation_fun

layernorm = partial(nn.LayerNorm, eps=1e-6)


###############################################################################
# Transformer
###############################################################################

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_sizes=[], activation="elu", stable_init=False):
        super().__init__()
        layers = []
        for i, size in enumerate(hidden_sizes):
            if i == 0:
                layers.append(nn.Linear(input_dim, size))
            else:
                layers.append(nn.Linear(hidden_sizes[i - 1], size))
            layers.append(get_activation_fun(activation))
        if len(hidden_sizes) > 0:
            layers.append(nn.Linear(hidden_sizes[-1], output_dim))
        else:
            layers.append(nn.Linear(input_dim, output_dim))
        self.net = nn.Sequential(*layers)
        # So-called stable init
        if stable_init:
            gain = 1.0 if output_dim == 1 else 0.01
            torch.nn.init.orthogonal_(self.net[-1].weight, gain=gain)

    def forward(self, x):
        return self.net(x)


class MLPBlock(nn.Module):
    """Transformer MLP block, fc, gelu, fc."""

    def __init__(self, w_in, mlp_d, dropout=0.0):
        super().__init__()
        self.linear_1 = nn.Linear(w_in, mlp_d, bias=True)
        self.dropout_1 = nn.Dropout(dropout, inplace=True)
        self.af = nn.GELU(approximate="tanh")
        self.linear_2 = nn.Linear(mlp_d, w_in, bias=True)
        self.dropout_2 = nn.Dropout(dropout, inplace=True)
        # Initialize the weights using xaiver uniform, biases using normal
        nn.init.xavier_uniform_(self.linear_1.weight)
        nn.init.xavier_uniform_(self.linear_2.weight)
        nn.init.normal_(self.linear_1.bias, std=1e-6)
        nn.init.normal_(self.linear_2.bias, std=1e-6)

    def forward(self, x):
        x = self.dropout_2(self.linear_2(self.af(self.dropout_1(self.linear_1(x)))))
        return x


class TransformerEncoderBlock(nn.Module):
    """Transformer encoder block, following https://arxiv.org/abs/2010.11929."""

    def __init__(self, hidden_d, n_heads, mlp_d, attn_dropout=0.0, mlp_dropout=0.0):
        super().__init__()
        self.ln_1 = layernorm(hidden_d)
        self.attention = MultiheadAttention(hidden_d, n_heads, dropout=attn_dropout, batch_first=True)
        self.ln_2 = layernorm(hidden_d)
        self.mlp_block = MLPBlock(hidden_d, mlp_d, dropout=mlp_dropout)

    def forward(self, x, attn_mask=None):
        x_p = self.ln_1(x)
        x_p, _ = self.attention(x_p, x_p, x_p, need_weights=False, attn_mask=attn_mask)
        x = x + x_p
        x_p = self.mlp_block(self.ln_2(x))
        return x + x_p


class TransformerEncoder(nn.Module):
    """Transformer encoder (sequence of TransformerEncoderBlock)."""

    def __init__(self, n_layers, hidden_d, n_heads, mlp_d, attn_dropout=0.0, mlp_dropout=0.0, last_layer_norm=False):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(n_layers):
            self.blocks.append(TransformerEncoderBlock(hidden_d, n_heads, mlp_d, attn_dropout, mlp_dropout))
        if last_layer_norm:
            self.ln = layernorm(hidden_d)

    def forward(self, x, attn_mask=None):
        for block in self.blocks:
            x = block(x, attn_mask=attn_mask)
        if hasattr(self, "ln"):
            x = self.ln(x)
        return x


def get_1d_sincos_pos_embed(embed_dim, length):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = np.arange(length, dtype=float)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class TransformerPolicy(nn.Module):
    def __init__(
        self,
        context_length,
        input_dim,
        output_dim,
        in_proj_hidden_sizes=[512],
        embed_dim=256,
        num_blocks=4,
        num_heads=4,
        mlp_ratio=2.,
        attn_dropout=0.0,
        mlp_dropout=0.0,
        last_layer_norm=False,
        head_hidden_sizes=[256],
    ):
        super().__init__()
        self.context_length = context_length

        self.in_proj = MLP(input_dim, embed_dim, in_proj_hidden_sizes, stable_init=False)
        pos_embed = get_1d_sincos_pos_embed(embed_dim, context_length)
        pos_embed = torch.from_numpy(pos_embed).float().unsqueeze(0)
        self.register_buffer("pos_embed", pos_embed)
        # self.pos_embed = nn.Parameter(torch.randn(1, context_length, embed_dim) * 0.02)
        self.encoder = TransformerEncoder(num_blocks, embed_dim, num_heads, int(embed_dim * mlp_ratio), attn_dropout, mlp_dropout, last_layer_norm)
        self.out_proj = MLP(embed_dim, output_dim, head_hidden_sizes, stable_init=True)

    def forward(self, observations, attn_masks=None):
        # observations: (context_length, batch_size, obs_dim)
        # attn_masks: None or (batch_size * n_heads, context_length, context_length)
        x = observations
        assert x.ndim == 3, "x must be (context_length, batch_size, input_dim)"
        if x.shape[0] != self.context_length:
            assert x.shape[0] > self.context_length, "x.shape[0] must be >= context_length"
            x = x[-self.context_length:]
            if attn_masks is not None:
                attn_masks = attn_masks[:, -self.context_length:, -self.context_length:]
        # In projection
        x = self.in_proj(x)
        # (context_length, batch_size, emb_dim) -> (batch_size, context_length, emb_dim)
        x = x.permute(1, 0, 2)
        # Add position embedding
        x = x + self.pos_embed
        # Transformer encoder
        x = self.encoder(x, attn_mask=attn_masks)
        # Output projection on the last step
        x = self.out_proj(x[:, -1])
        return x

    def compute_features(self, observations, attn_masks=None):
        # observations: (context_length, batch_size, obs_dim)
        # attn_masks: None or (batch_size * n_heads, context_length, context_length)
        x = observations
        assert x.ndim == 3, "x must be (context_length, batch_size, input_dim)"
        assert x.shape[0] == self.context_length
        # In projection
        x = self.in_proj(x)
        # (context_length, batch_size, emb_dim) -> (batch_size, context_length, emb_dim)
        x = x.permute(1, 0, 2)
        # Add position embedding
        x = x + self.pos_embed
        # Transformer encoder
        x = self.encoder(x, attn_mask=attn_masks)
        # Save features from the encoder
        features = x.detach().clone()
        # Output projection on the last step
        x = self.out_proj(x[:, -1])
        return x, features


##################################################
# Diffusion policy head
##################################################

class DiffusionPolicyHead(nn.Module):
    def __init__(
        self,
        context_length,
        input_dim,
        input_condition_dim,
        output_dim,
        num_diffusion_steps,
        embed_dim=128,
        num_blocks=4,
        num_heads=4,
        mlp_ratio=2.,
        attn_dropout=0.0,
        mlp_dropout=0.0,
        last_layer_norm=False,
    ):
        super().__init__()
        self.context_length = context_length

        self.in_proj = nn.Linear(input_dim, embed_dim)
        self.in_cond_proj = nn.Linear(input_condition_dim, embed_dim)

        pos_embed = get_1d_sincos_pos_embed(embed_dim, context_length)
        pos_embed = torch.from_numpy(pos_embed).float().unsqueeze(0)
        self.register_buffer("pos_embed", pos_embed)

        time_embed = get_1d_sincos_pos_embed(embed_dim, num_diffusion_steps)
        time_embed = torch.from_numpy(time_embed).float().unsqueeze(1)
        self.register_buffer("time_embed", time_embed)

        self.encoder = TransformerEncoder(num_blocks, embed_dim, num_heads, int(embed_dim * mlp_ratio), attn_dropout, mlp_dropout, last_layer_norm)
        self.out_proj = nn.Linear(embed_dim, output_dim)

    def forward(self, observations, timesteps, conditions, attn_masks=None):
        # observations: (context_length, batch_size, obs_dim)
        # attn_masks: None or (batch_size * n_heads, context_length, context_length)
        x = observations
        assert x.ndim == 3, "x must be (batch_size, context_length, input_dim)"
        assert conditions.ndim == 3, "conditions should be (batch_size, context_length, condition_dim)"
        assert timesteps.ndim == 1, "Diffusion time steps must be (batch_size, )"
        # In projection
        x = self.in_proj(x)
        conditions = self.in_cond_proj(conditions)
        # add condition to x
        x = x + conditions
        # Add position embedding
        x = x + self.pos_embed
        # Add diffusion timestep embedding
        x = x + self.time_embed[timesteps]
        # Transformer encoder
        x = self.encoder(x, attn_mask=attn_masks)
        # Output projection on the last step
        x = self.out_proj(x)
        return x

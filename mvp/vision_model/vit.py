#!/usr/bin/env python3

"""Vision Transformer (ViT) implementation (adapted from MAE and timm)."""

import os
import timm.models.vision_transformer
import numpy as np

from functools import partial

import torch
import torch.nn as nn

try:
    from s2wrapper import forward as multiscale_forward
except ImportError:
    multiscale_forward = None


def get_1d_sincos_pos_embed(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):

    def __init__(self, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)
        # remove the classifier
        if hasattr(self, 'pre_logits'):
            del self.pre_logits
        if hasattr(self, 'head'):
            del self.head

    def extract_feat(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed

        for blk in self.blocks:
            x = blk(x)

        return x

    def forward_norm(self, x):
        return self.norm(x)

    def forward_feature(self, x):
        return self.forward_norm(self.extract_feat(x))

    def forward(self, x, mode='cls'):
        if 'multiscale' in mode:
            x = multiscale_forward(self.forward_feature, x, scales=[1, 2], num_prefix_token=1)
        else:
            x = self.forward_feature(x)

        if 'cls' in mode:
            x = x[:, 0].detach().float()
        elif 'mean' in mode:
            x = x[:, 1:].mean(dim=1).detach().float()
        elif 'all' in mode:
            x = x.detach().float()
        else:
            raise NotImplementedError
        return x

    def freeze(self):
        self.pos_embed.requires_grad = False
        self.cls_token.requires_grad = False

        def _freeze_module(m):
            for p in m.parameters():
                p.requires_grad = False

        _freeze_module(self.patch_embed)
        _freeze_module(self.blocks)
        _freeze_module(self.norm)

    def trainable_param_names(self):
        trainable_params = []
        for name, p in self.named_parameters():
            if p.requires_grad:
                trainable_params.append(name)
        return trainable_params


def vit_s16(pretrained, **kwargs):
    scratch = pretrained.endswith("none")
    assert scratch or os.path.exists(pretrained)
    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    # load from checkpoint
    if not scratch:
        if pretrained.endswith("pyth"):
            load_raw_checkpoint(pretrained, model)
        else:
            load_checkpoint(pretrained, model)
        print("Loaded encoder from: {}".format(pretrained))
    hidden_dim = 384
    return model, hidden_dim


def vit_b16(pretrained, **kwargs):
    scratch = pretrained.endswith("none")
    assert scratch or os.path.exists(pretrained)
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    # load from checkpoint
    if not scratch:
        if pretrained.endswith("pyth"):
            load_raw_checkpoint(pretrained, model)
        else:
            load_checkpoint(pretrained, model)
        print("Loaded encoder from: {}".format(pretrained))
    hidden_dim = 768
    return model, hidden_dim


def vit_l16(pretrained, **kwargs):
    scratch = pretrained.endswith("none")
    assert scratch or os.path.exists(pretrained)
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    # load from checkpoint
    if not scratch:
        if pretrained.endswith("pyth"):
            load_raw_checkpoint(pretrained, model)
        else:
            load_checkpoint(pretrained, model)
        print("Loaded encoder from: {}".format(pretrained))
    hidden_dim = 1024
    return model, hidden_dim


def vit_h16(pretrained, **kwargs):
    scratch = pretrained.endswith("none")
    assert scratch or os.path.exists(pretrained)
    model = VisionTransformer(
        patch_size=16, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    # load from checkpoint
    if not scratch:
        if pretrained.endswith("pyth"):
            load_raw_checkpoint(pretrained, model)
        else:
            load_checkpoint(pretrained, model)
        print("Loaded encoder from: {}".format(pretrained))
    hidden_dim = 1280
    return model, hidden_dim


def unwrap_model(model):
    """Remove the DistributedDataParallel wrapper if present."""
    wrapped = isinstance(model, torch.nn.parallel.distributed.DistributedDataParallel)
    return model.module if wrapped else model


def load_checkpoint(checkpoint_file, model):
    """Loads a checkpoint selectively based on the input options."""
    assert os.path.exists(checkpoint_file)
    checkpoint = torch.load(checkpoint_file, map_location="cpu")
    state_dict = checkpoint["model"]
    r = unwrap_model(model).load_state_dict(state_dict, strict=False)
    if r.unexpected_keys or r.missing_keys:
        print(f"Loading weights, unexpected keys: {r.unexpected_keys}")
        print(f"Loading weights, missing keys: {r.missing_keys}")


def load_raw_checkpoint(checkpoint_file, model):
    """Loads a raw pycls checkpoint selectively based on the input options."""
    assert os.path.exists(checkpoint_file)
    checkpoint = torch.load(checkpoint_file, map_location="cpu")
    state_dict = checkpoint["model_state"]
    state_dict["pos_embed"] = torch.cat((torch.zeros(1, 1, model.embed_dim), state_dict["pos_embed"]), dim=1)
    state_dict= {k: v for k, v in state_dict.items() if "decoder" not in k}
    del state_dict["mask_embed"]
    del state_dict["patch_predictor.head_fc.weight"]
    del state_dict["patch_predictor.head_fc.bias"]
    state_dict["patch_embed.proj.weight"] = state_dict.pop("patch_embed.weight").reshape(model.embed_dim, 3, *model.patch_embed.patch_size)
    state_dict["patch_embed.proj.bias"] = state_dict.pop("patch_embed.bias")
    r = unwrap_model(model).load_state_dict(state_dict, strict=False)
    if r.unexpected_keys or r.missing_keys:
        print(f"Loading weights, unexpected keys: {r.unexpected_keys}")
        print(f"Loading weights, missing keys: {r.missing_keys}")

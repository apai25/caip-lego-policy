#!/usr/bin/env python3

"""Vision Transformer (ViT) implementation (adapted from MAE and timm)."""

import os
import timm.models.vision_transformer
import numpy as np

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

def get_1d_sincos_pos_embed(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float)
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
        del self.pre_logits, self.head
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
        # sam model
        sam = sam_model_registry["vit_l"](checkpoint="/shared/bfshi/projects/mmsegmentation/pretrain/sam_vit_l_0b3195.pth")
        self.mask_generator = SamAutomaticMaskGenerator(sam)

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

    def forward(self, x, mode='cls', x_for_sam=None):
        assert x_for_sam is not None
        x = self.forward_norm(self.extract_feat(x))
        masks = self.mask_generator.generate(x_for_sam)
        masks.sort(key=lambda x: x['area'], reverse=True)
        masks = [torch.Tensor(item['segmentation']).to(torch.float).to(x.device) for item in masks]
        masks = torch.stack(masks).unsqueeze(1)  # n_mask*1*H*W
        n_masks = masks.shape[0]
        if n_masks < 50:
            masks = torch.cat([masks, torch.ones((50-n_masks, 1, masks.shape[2], masks.shape[3]), device=masks.device)])
        elif n_masks > 50:
            masks = masks[:50]
            n_masks = 50

        H = W = int((x.shape[1] - 1)**0.5)
        assert H*W == x.shape[1]-1
        x = x[:, 1:].reshape(1, H, W, -1).permute(0, 3, 1, 2)
        x = F.interpolate(x, size=(masks.shape[1], masks.shape[2]), mode='bilinear')  # 1*C*H*W

        masked_features = (x * masks).sum(dim=(-2, -1)) / masks.sum(dim=(-2, -1))
        if mode == 'mean':
            x = masked_features[:n_masks].mean(dim=0)[None].detach().float()
        elif mode == 'all':
            x = masked_features[None].detach().float()
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


def vit_s16_sam_mask(pretrained, **kwargs):
    scratch = pretrained.endswith("none")
    assert scratch or os.path.exists(pretrained)
    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    # load from checkpoint
    if not scratch:
        load_checkpoint(pretrained, model)
        print("Loaded encoder from: {}".format(pretrained))
    hidden_dim = 384
    return model, hidden_dim


def vit_b16_sam_mask(pretrained, **kwargs):
    scratch = pretrained.endswith("none")
    assert scratch or os.path.exists(pretrained)
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    # load from checkpoint
    if not scratch:
        load_checkpoint(pretrained, model)
        print("Loaded encoder from: {}".format(pretrained))
    hidden_dim = 768
    return model, hidden_dim


def vit_l16_sam_mask(pretrained, **kwargs):
    scratch = pretrained.endswith("none")
    assert scratch or os.path.exists(pretrained)
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    # load from checkpoint
    if not scratch:
        load_checkpoint(pretrained, model)
        print("Loaded encoder from: {}".format(pretrained))
    hidden_dim = 1024
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

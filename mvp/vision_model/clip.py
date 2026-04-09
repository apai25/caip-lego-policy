#!/usr/bin/env python3

"""CLIP wrapper (adapted from CLIP)."""

import os
import numpy as np

import torch
import torch.nn as nn

import clip


class ClipModel(nn.Module):

    def __init__(self, name, device):
        super().__init__()
        self.model, _ = clip.load(name, device=device)
        del self.model.transformer
        self.model.float()

    def extract_feat(self, x):
        #x = self.model.encode_image(x)
        x = self.model.encode_image(x).float()
        return x

    def forward_norm(self, x):
        # norm in encode_image
        return x

    def forward(self, x):
        return self.forward_norm(self.extract_feat(x))

    def freeze(self):
        for p in self.model.parameters():
            p.requires_grad = False


def resnet50():
    name = "RN50"
    device = torch.cuda.current_device()
    model = ClipModel(name, device)
    hidden_dim = 1024
    return model, hidden_dim


def vit_b16():
    name = "ViT-B/16"
    device = torch.cuda.current_device()
    model = ClipModel(name, device)
    hidden_dim = 512
    return model, hidden_dim

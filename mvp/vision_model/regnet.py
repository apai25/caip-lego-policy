#!/usr/bin/env python3

"""RegNet (torchvision wrapper)."""

import os
import torchvision
import numpy as np

import torch
import torch.nn as nn


class RegNet(nn.Module):

    def __init__(self, model):
        super(RegNet, self).__init__()
        # original model
        self.model = model
        # layer norm
        self.norm = nn.LayerNorm(self.model.fc.in_features, eps=1e-6)
        # remove the classifier
        del self.model.fc

    def extract_feat(self, x):
        x = self.model.stem(x)
        x = self.model.trunk_output(x)

        x = self.model.avgpool(x)
        x = x.flatten(start_dim=1)

        return x

    def forward_norm(self, x):
        return self.norm(x)

    def forward(self, x):
        return self.forward_norm(self.extract_feat(x))

    def freeze(self):
        for p in self.model.parameters():
            p.requires_grad = False

        trainable_params = []
        for name, p in self.named_parameters():
            if p.requires_grad:
                trainable_params.append(name)

        print("Trainable parameters in the encoder:")
        print(trainable_params)


def regnety800mf(pretrained):
    model = RegNet(torchvision.models.regnet_y_800mf(pretrained=pretrained))
    hidden_dim = 384
    return model, hidden_dim


def regnety3_2gf(pretrained):
    model = RegNet(torchvision.models.regnet_y_3_2gf(pretrained=pretrained))
    hidden_dim = 1512
    return model, hidden_dim


def regnety8gf(pretrained):
    model = RegNet(torchvision.models.regnet_y_8gf(pretrained=pretrained))
    hidden_dim = 2016
    return model, hidden_dim

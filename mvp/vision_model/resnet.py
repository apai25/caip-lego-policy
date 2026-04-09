#!/usr/bin/env python3

"""ResNet (torchvision wrapper)."""

import os
import torchvision
import numpy as np

import torch
import torch.nn as nn


class ResNet(nn.Module):

    def __init__(self, model):
        super(ResNet, self).__init__()
        # original model
        self.model = model
        # layer norm
        self.norm = nn.LayerNorm(self.model.fc.in_features, eps=1e-6)
        # remove the classifier
        del self.model.fc

    def extract_feat(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)

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


def resnet18(pretrained):
    model = ResNet(torchvision.models.resnet18(pretrained=pretrained))
    hidden_dim = 512
    return model, hidden_dim


def resnet50(pretrained):
    model = ResNet(torchvision.models.resnet50(pretrained=pretrained))
    hidden_dim = 2048
    return model, hidden_dim

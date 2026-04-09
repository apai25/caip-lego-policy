import os
import torch.nn as nn
from mvp.vision_model import vit


# Available models
_MODELS = {
    "vits-mae-hoi": "mae_pretrain_hoi_vit_small.pth",
    "vits-mae-egosoup": "mae_pretrain_egosoup_vit_small.pyth",
    "vitb-mae-egosoup": "mae_pretrain_egosoup_vit_base.pth",
    "vitl-mae-egosoup": "mae_pretrain_egosoup_vit_large_256.pth",
    "vith-mae-egosoup": "mae_pretrain_egosoup_vit_huge_256.pyth",
}


class Encoder(nn.Module):

    def __init__(self, model_name, pretrain_dir, freeze):
        super(Encoder, self).__init__()
        assert model_name in _MODELS
        if model_name == "vits-mae-hoi":
            pretrain_path = os.path.join(pretrain_dir, _MODELS[model_name])
            self.backbone, self.emb_dim = vit.vit_s16(pretrain_path)
        elif model_name == "vits-mae-egosoup":
            pretrain_path = os.path.join(pretrain_dir, _MODELS[model_name])
            self.backbone, self.emb_dim = vit.vit_s16(pretrain_path)
        elif model_name == "vitb-mae-egosoup":
            pretrain_path = os.path.join(pretrain_dir, _MODELS[model_name])
            self.backbone, self.emb_dim = vit.vit_b16(pretrain_path)
        elif model_name == "vitl-mae-egosoup":
            pretrain_path = os.path.join(pretrain_dir, _MODELS[model_name])
            self.backbone, self.emb_dim = vit.vit_l16(pretrain_path, img_size=256)
        elif model_name == "vith-mae-egosoup":
            pretrain_path = os.path.join(pretrain_dir, _MODELS[model_name])
            self.backbone, self.emb_dim = vit.vit_h16(pretrain_path, img_size=256)
        else:
            raise NotImplementedError
        if freeze:
            self.backbone.freeze()
        print("Trainable encoder parameters:")
        print(self.backbone.trainable_param_names())

    def forward(self, x, mode='cls', **kwargs):
        return self.backbone(x, mode, **kwargs)

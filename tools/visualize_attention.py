#!/usr/bin/env python3

"""Behavior cloning (BC)."""

import hydra
import omegaconf
import numpy as np
import os
import random
import statistics
import time
import wandb
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from torch.utils.data.distributed import DistributedSampler

from mvp.utils.sys_utils import omegaconf_to_dict, print_dict, dump_cfg
from mvp.utils.sys_utils import set_np_formatting, set_seed
from mvp.bimanual_bc.actor import ActorTransformerConcatAttnPooling
from mvp.bimanual_bc.dataset import Bimanual_Dataset

from tools.store_mae_features import Encoder


@torch.no_grad()
def run_model(cfg, test_loader, model, vision_encoder, cur_epoch):
    # Enable eval mode
    model.eval()
    for cur_iter, items in enumerate(test_loader):
        ims, pi_obs, pi_act, prompts, visible_cam_mask, mod_mask, att_mask, img_selected_ids = items
        ims, pi_obs, pi_act, prompts, visible_cam_mask, mod_mask, att_mask, img_selected_ids = \
            [ims_c.cuda() for ims_c in
             ims], pi_obs.cuda(), pi_act.cuda(), prompts.cuda(), visible_cam_mask.cuda(), mod_mask.cuda(), att_mask.cuda(), img_selected_ids.cuda()
        ims = [ims_c.to(torch.float32) for ims_c in ims]

        B, T = ims[0].shape[:2]
        img_features = [vision_encoder(rearrange(ims_c, 'b t c h w -> (b t) c h w')) for ims_c in ims]
        img_features = [rearrange(img_features_c, '(b t) n c -> b t n c', b=B, t=T) for img_features_c in img_features]

        # concat image feature into pi_obs, also expand mod_mask
        attentions = model.vis_att(im_features=img_features,
                                   prompts=prompts[torch.arange(img_selected_ids.shape[0]).unsqueeze(1), img_selected_ids],
                                   cam_ids=list(range(len(cfg.data.cams))))

        # log the first image in the batch and its attention maps for each cam and each head
        for cam_id in range(len(cfg.data.cams)):
            image = ims[cam_id][0, 0].permute(1, 2, 0)
            image = image.detach().cpu().numpy()
            image = wandb.Image(image, caption=f"image_cam{cam_id}")
            wandb.log({"Input image": image})

            for head_id in range(len(attentions[0])):
                att = attentions[cam_id][head_id][0]
                att = att.detach().cpu().numpy()
                att = wandb.Image(att[..., None], caption=f"attention_cam{cam_id}_head{head_id}")
                wandb.log({"Attention maps": att})


def visualize_attention(cfg):
    """Trains a model with BC."""

    # Construct test dataset/loader
    assert cfg.data.num_test > 0
    assert not cfg.data.features
    assert cfg.data.inmem
    random.seed(cfg.seed)
    test_dataset = Bimanual_Dataset(
        features=cfg.data.features,
        demo_root=cfg.data.demo_root,
        demo_dirs=cfg.data.demo_dirs,
        inmem=cfg.data.inmem,
        start_ind=cfg.data.offset + cfg.data.num_train,
        num_demos=cfg.data.num_test,
        num_steps=cfg.actor.num_steps + cfg.actor.num_pred,
        num_pred=cfg.actor.num_pred,
        look_ahead=cfg.actor.look_ahead,
        im_size=cfg.data.im_size,
        cams=cfg.data.cams,
        noisy_skip=cfg.data.noisy_skip,
        frame_skip=cfg.data.frame_skip,
        default_pos_left_arm=cfg.data.default_pos_left_arm,
        default_pos_right_arm=cfg.data.default_pos_right_arm,
        joint_noise_mean=[0.] * 24,
        joint_noise_std=[0.] * 24,
        joint_noise_std_scale=cfg.data.joint_noise_std_scale,
        data_filter=cfg.data.data_filter,
        history_repeating=cfg.data.history_repeating,
        img_sample_num=cfg.data.img_sample_num,
        use_all_features=False,
        action_data_ratio=cfg.data.action_data_ratio,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=(cfg.train.mb_size // cfg.num_gpus),
        shuffle=False,
        num_workers=4, pin_memory=True, drop_last=False
    )

    # Construct the model
    assert "transformer_concat_attnpool"
    model = ActorTransformerConcatAttnPooling(
        context_length=cfg.actor.num_steps,
        obs_shape=cfg.actor.obs_dim,
        actions_shape=cfg.actor.act_dim,
        num_pred=cfg.actor.num_pred,
        policy_cfg=cfg.transformer_concat,
        normalize_input=cfg.transformer_concat.normalize_input,
        normalize_bn_cls=nn.BatchNorm1d,
        num_cam=len(cfg.data.cams),
        im_dim=cfg.actor.im_dim,
        prompt_dim=cfg.actor.prompt_dim,
    )

    print("Num of params: ", sum(p.numel() for p in model.parameters()) / 1e6)

    # Load a checkpoint
    if cfg.test.weights:
        state_dict = torch.load(cfg.test.weights, map_location="cpu")["model_state"]
        mks, uks = model.load_state_dict(state_dict, strict=False)
        print("loaded checkpoint from: {}".format(cfg.train.weights))
        print("missing key: {}".format(mks))
        print("unexpected keys: {}".format(uks))

    # Transfer the model to the gpu
    model = model.cuda()

    # initialize the vision encoder
    vision_encoder = Encoder(cfg.encoder.name, cfg.encoder.pretrain_dir, freeze=True)
    vision_encoder = vision_encoder.cuda()
    vision_encoder = partial(vision_encoder, mode=cfg.encoder.save_mode)

    assert cfg.data.num_test > 0, "test-only mode needs data.num_test > 0"
    run_model(cfg, test_loader, model, vision_encoder, 0)


@hydra.main(version_base=None, config_name="config", config_path="../configs/bimanual_bc")
def main(cfg: omegaconf.DictConfig):

    assert cfg.num_gpus == 1

    # Set up logging
    print_dict(omegaconf_to_dict(cfg))
    wandb.init(
        dir=cfg.logdir,
        name=cfg.wandb.name,
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        config=omegaconf.OmegaConf.to_container(cfg),
        mode=cfg.wandb.mode,
    )

    # Set rng seed
    seed = cfg.seed
    set_np_formatting()
    set_seed(seed)

    # Perform training
    visualize_attention(cfg)


if __name__ == '__main__':
    main()
#!/usr/bin/env python3

"""Behavior cloning (BC)."""

import numpy as np
import os
import random
import statistics
import time
import wandb

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from mvp.bimanual_bc.actor import ActorMLP
from mvp.bimanual_bc.actor import ActorTransformerConcat_Bimanual, \
    ActorTransformerConcatAttnPooling_Bimanual, \
    ActorTransformerConcatMeanPooling_Bimanual, \
    ActorTransformerConcatAttnPoolingDiffusion_Bimanual
from mvp.bimanual_bc.dataset import Bimanual_Dataset, BKL_Dataset
from mvp.utils.meters import TrainMeter, TestMeter
from mvp.utils.utils import adjust_lr
from mvp.utils.utils import get_optimizer_groups
from mvp.utils.utils import save_checkpoint
from mvp.utils.utils import scaled_all_reduce
from mvp.utils.utils import masked_mse_loss, masked_l1_loss, masked_ce_loss


@torch.no_grad()
def test_epoch(cfg, test_loader, model, meter, cur_epoch):
    # Enable eval mode
    model.eval()
    meter.reset()
    meter.iter_tic()
    for cur_iter, items in enumerate(test_loader):
        ims, pi_obs, pi_obs_noiseless, pi_act, prompts, prompts_text, visible_cam_mask, mod_mask, att_mask, img_selected_ids = items
        ims, pi_obs, pi_obs_noiseless, pi_act, prompts, visible_cam_mask, mod_mask, att_mask, img_selected_ids = \
            [ims_c.cuda() for ims_c in ims], pi_obs.cuda(), pi_obs_noiseless.cuda(), pi_act.cuda(), prompts.cuda(), visible_cam_mask.cuda(), mod_mask.cuda(), att_mask.cuda(), img_selected_ids.cuda()
        ims = [ims_c.to(torch.float32) for ims_c in ims]
        prompts_text = list(zip(*prompts_text))  # list of B elements, each element is a list of T strings
        assert len(prompts_text) == pi_obs.shape[0] and len(prompts_text[0]) == pi_obs.shape[1]

        # construct attention mask
        att_mask = att_mask[:, :cfg.actor.num_steps].unsqueeze(1).repeat(1, cfg.actor.num_steps, 1)  # B * L * L
        att_mask[:, range(cfg.actor.num_steps), range(cfg.actor.num_steps)] = 0  # diagonal elements must be 0
        B, L, L = att_mask.shape
        att_mask = att_mask.unsqueeze(1).repeat(1, cfg.transformer_concat.num_heads, 1, 1).view(B * cfg.transformer_concat.num_heads, L, L)
        att_mask = att_mask.bool()

        # construct causal mask
        if cfg.transformer_concat.causal:
            causal_mask = torch.triu(torch.ones((cfg.actor.num_steps, cfg.actor.num_steps), device=pi_obs.device), diagonal=1).bool()
            causal_mask = causal_mask.unsqueeze(0).repeat(att_mask.shape[0], 1, 1)
            att_mask = torch.logical_or(att_mask, causal_mask)

        # concat image feature into pi_obs, also expand mod_mask
        if 'attnpool' in cfg.actor.type or 'meanpool' in cfg.actor.type:
            ims = model(im_features=ims,
                        prompts=prompts[torch.arange(img_selected_ids.shape[0]).unsqueeze(1), img_selected_ids],
                        cam_ids=list(range(len(cfg.data.cams))),
                        attn_pool_only=True)
        elif ims[0].dim() == 4:
            for i, ims_c in enumerate(ims):
                ims_c = ims_c[:, :, 1:]
                B, T, N, C = ims_c.shape
                ims_c = ims_c.reshape(B*T, int(N**0.5), int(N**0.5), C)  # (B*T, H, W, C)
                ims_c = F.avg_pool2d(ims_c.permute(0, 3, 1, 2), kernel_size=cfg.data.img_downsample).permute(0, 2, 3, 1)  # (B*T, H/2, W/2, C)
                ims_c = ims_c.reshape(B, T, N // (cfg.data.img_downsample ** 2), C)
                ims[i] = ims_c
        all_ims = [torch.zeros(ims_c.shape[0], pi_obs.shape[1], *ims_c.shape[2:]).cuda() for ims_c in ims]  # ims_c could be (B, T, C) or (B, T, N, C)
        for all_ims_c, ims_c in zip(all_ims, ims):
            all_ims_c[torch.arange(img_selected_ids.shape[0]).unsqueeze(1), img_selected_ids] = ims_c
        obs_each_mod = [prompts] + all_ims + [pi_obs]

        # prepare mask for each modality
        prompt_mask = torch.ones_like(prompts).cuda()
        img_masks = [torch.zeros(all_ims_c.shape[0], pi_obs.shape[1], all_ims_c.shape[-1]).cuda() for all_ims_c in all_ims]
        cam_masks = visible_cam_mask.split(1, dim=-1)
        for img_mask, cam_mask in zip(img_masks, cam_masks):
            img_mask[torch.arange(img_selected_ids.shape[0]).unsqueeze(1), img_selected_ids] = 1
            img_mask = img_mask * cam_mask[..., None]
        _act_dim = pi_act.shape[-1]
        state_mask = mod_mask[:, :, :-_act_dim]
        action_mask = mod_mask[:, :, -_act_dim:]
        if not getattr(cfg.data, 'use_proprio', True):
            state_mask = torch.zeros_like(state_mask)
        mask_each_mod = [prompt_mask] + img_masks + [state_mask]

        # model feedforward
        preds_each_mod = model(obs_each_mod=[obs[:, :cfg.actor.num_steps] for obs in obs_each_mod],
                               mask_each_mod=[mask[:, :cfg.actor.num_steps] for mask in mask_each_mod],
                               attn_mask=att_mask,
                               img_selected_ids=img_selected_ids[:, :cfg.data.img_sample_num])

        # prediction loss
        targets_each_mod = [obs[:, -cfg.actor.num_pred:] for obs in obs_each_mod[:-1]] + [pi_obs_noiseless[:, -cfg.actor.num_pred:]] + [pi_act[:, -cfg.actor.num_pred-1:-1]]
        loss_mask_each_mod = [mask[:, -cfg.actor.num_pred:] for mask in mask_each_mod] + [action_mask[:, -cfg.actor.num_pred - 1:-1]]
        use_ce_loss_each_mod = [False for _ in range(len(targets_each_mod))]
        loss_each_mod = [masked_l1_loss(preds, targets, mask) if not use_ce_loss else masked_ce_loss(preds, targets, mask)
                         for preds, targets, mask, use_ce_loss in zip(preds_each_mod, targets_each_mod, loss_mask_each_mod, use_ce_loss_each_mod)]

        if 'diffusion' in cfg.actor.type:
            loss_each_mod[-1] = model.module.compute_diffusion_loss(preds_each_mod[-1], targets_each_mod[-1], loss_mask_each_mod[-1])

        prompt_loss = loss_each_mod[0]
        num_cams = len(cfg.data.cams)
        img_losses = loss_each_mod[1:1 + num_cams]
        img_loss = sum(img_losses) / num_cams
        state_loss = loss_each_mod[-2]
        action_loss = loss_each_mod[-1]
        print('##########')
        print('Prompt loss:', prompt_loss)
        for i, cam in enumerate(cfg.data.cams):
            print(f'Image ({cam}) loss:', img_losses[i])
        print('State loss:', state_loss)
        print('Action loss:', action_loss)
        print('##########')

        loss = cfg.actor.prompt_loss_weight * prompt_loss + 0 * state_loss + cfg.actor.state_loss_weight * img_loss + action_loss

        # Sync stats across GPUs
        loss = scaled_all_reduce([loss])[0].item()
        # Record the stats
        meter.iter_toc()
        meter.update_stats(loss, pi_obs.shape[0] * cfg.num_gpus)
        # Log iter stats
        meter.log_iter_stats(cur_epoch, cur_iter)
        meter.iter_tic()
    # Log epoch stats
    meter.log_epoch_stats(cur_epoch)


def train_epoch(cfg, train_loader, model, optimizer, meter, cur_epoch):
    # Enable training mode
    model.train()
    meter.reset()
    meter.iter_tic()
    for cur_iter, items in enumerate(train_loader):
        ims, pi_obs, pi_obs_noiseless, pi_act, prompts, prompts_text, visible_cam_mask, mod_mask, att_mask, img_selected_ids = items
        ims, pi_obs, pi_obs_noiseless, pi_act, prompts, visible_cam_mask, mod_mask, att_mask, img_selected_ids = \
            [ims_c.cuda() for ims_c in ims], pi_obs.cuda(), pi_obs_noiseless.cuda(), pi_act.cuda(), prompts.cuda(), visible_cam_mask.cuda(), mod_mask.cuda(), att_mask.cuda(), img_selected_ids.cuda()
        ims = [ims_c.to(torch.float32) for ims_c in ims]
        prompts_text = list(zip(*prompts_text))  # list of B elements, each element is a list of T strings
        assert len(prompts_text) == pi_obs.shape[0] and len(prompts_text[0]) == pi_obs.shape[1]

        # Udapte the learning rate
        lr = adjust_lr(
            optimizer, cfg.train.lr, cur_epoch + float(cur_iter) / len(train_loader),
            cfg.train.warmup_ep, cfg.train.num_ep
        )

        # construct attention mask
        att_mask = att_mask[:, :cfg.actor.num_steps].unsqueeze(1).repeat(1, cfg.actor.num_steps, 1)  # B * L * L
        att_mask[:, range(cfg.actor.num_steps), range(cfg.actor.num_steps)] = 0  # diagonal elements must be 0
        B, L, L = att_mask.shape
        att_mask = att_mask.unsqueeze(1).repeat(1, cfg.transformer_concat.num_heads, 1, 1).view(B * cfg.transformer_concat.num_heads, L, L)
        att_mask = att_mask.bool()

        # construct causal mask
        if cfg.transformer_concat.causal:
            causal_mask = torch.triu(torch.ones((cfg.actor.num_steps, cfg.actor.num_steps), device=pi_obs.device), diagonal=1).bool()
            causal_mask = causal_mask.unsqueeze(0).repeat(att_mask.shape[0], 1, 1)
            att_mask = torch.logical_or(att_mask, causal_mask)

        # prepare input of each modality
        if 'attnpool' in cfg.actor.type or 'meanpool' in cfg.actor.type:
            ims = model(im_features=ims,
                        prompts=prompts[torch.arange(img_selected_ids.shape[0]).unsqueeze(1), img_selected_ids],
                        cam_ids=list(range(len(cfg.data.cams))),
                        attn_pool_only=True)
        elif ims[0].dim() == 4:
            for i, ims_c in enumerate(ims):
                ims_c = ims_c[:, :, 1:]
                B, T, N, C = ims_c.shape
                ims_c = ims_c.reshape(B*T, int(N**0.5), int(N**0.5), C)  # (B*T, H, W, C)
                ims_c = F.avg_pool2d(ims_c.permute(0, 3, 1, 2), kernel_size=cfg.data.img_downsample).permute(0, 2, 3, 1)  # (B*T, H/2, W/2, C)
                ims_c = ims_c.reshape(B, T, N // (cfg.data.img_downsample ** 2), C)
                ims[i] = ims_c

        all_ims = [torch.zeros(ims_c.shape[0], pi_obs.shape[1], *ims_c.shape[2:]).cuda() for ims_c in ims]  # ims_c could be (B, T, C) or (B, T, N, C)
        for all_ims_c, ims_c in zip(all_ims, ims):
            all_ims_c[torch.arange(img_selected_ids.shape[0]).unsqueeze(1), img_selected_ids] = ims_c
        obs_each_mod = [prompts] + all_ims + [pi_obs]

        # prepare mask for each modality
        prompt_mask = torch.ones_like(prompts).cuda()
        img_masks = [torch.zeros(all_ims_c.shape[0], pi_obs.shape[1], all_ims_c.shape[-1]).cuda() for all_ims_c in all_ims]
        cam_masks = visible_cam_mask.split(1, dim=-1)
        for img_mask, cam_mask in zip(img_masks, cam_masks):
            img_mask[torch.arange(img_selected_ids.shape[0]).unsqueeze(1), img_selected_ids] = 1
            img_mask = img_mask * cam_mask[..., None]
        _act_dim = pi_act.shape[-1]
        state_mask = mod_mask[:, :, :-_act_dim]
        action_mask = mod_mask[:, :, -_act_dim:]
        # Zero out proprio mask if use_proprio is False
        if not getattr(cfg.data, 'use_proprio', True):
            state_mask = torch.zeros_like(state_mask)
        mask_each_mod = [prompt_mask] + img_masks + [state_mask]

        # model feedforward
        preds_each_mod = model(obs_each_mod=[obs[:, :cfg.actor.num_steps] for obs in obs_each_mod],
                               mask_each_mod=[mask[:, :cfg.actor.num_steps] for mask in mask_each_mod],
                               attn_mask=att_mask,
                               img_selected_ids=img_selected_ids[:, :cfg.data.img_sample_num])

        # prediction loss
        targets_each_mod = [obs[:, -cfg.actor.num_pred:] for obs in obs_each_mod[:-1]] + [pi_obs_noiseless[:, -cfg.actor.num_pred:]] + [pi_act[:, -cfg.actor.num_pred-1:-1]]
        loss_mask_each_mod = [mask[:, -cfg.actor.num_pred:] for mask in mask_each_mod] + [action_mask[:, -cfg.actor.num_pred-1:-1]]
        use_ce_loss_each_mod = [False for _ in range(len(targets_each_mod))]
        loss_each_mod = [masked_l1_loss(preds, targets, mask) if not use_ce_loss else masked_ce_loss(preds, targets, mask)
                         for preds, targets, mask, use_ce_loss in zip(preds_each_mod, targets_each_mod, loss_mask_each_mod, use_ce_loss_each_mod)]

        if 'diffusion' in cfg.actor.type:
            loss_each_mod[-1] = model.module.compute_diffusion_loss(preds_each_mod[-1], targets_each_mod[-1], loss_mask_each_mod[-1])

        prompt_loss = loss_each_mod[0]
        img_loss = sum(loss_each_mod[1: 1 + len(all_ims)]) / len(all_ims)
        state_loss = loss_each_mod[-2]
        action_loss = loss_each_mod[-1]

        if cfg.actor.pred_action_only:
            prompt_loss *= 0
            img_loss *= 0
            state_loss *= 0
        if cfg.actor.pred_state_only:
            action_loss *= 0
        loss = cfg.actor.prompt_loss_weight * prompt_loss + 0 * state_loss + cfg.actor.state_loss_weight * img_loss + action_loss

        # Compute the gradients
        optimizer.zero_grad()
        loss.backward()
        if cfg.train.clip_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.clip_norm)
        # Update the parameters
        optimizer.step()
        # Sync stats across GPUs
        loss = scaled_all_reduce([loss])[0].item()
        # Record the stats
        meter.iter_toc()
        meter.update_stats(loss, lr, pi_obs.shape[0] * cfg.num_gpus)
        # Log iter stats
        meter.log_iter_stats(cur_epoch, cur_iter)
        meter.iter_tic()
    # Log epoch stats
    meter.log_epoch_stats(cur_epoch)


def train(cfg):
    """Trains a model with BC."""

    # Construct train dataset/loader
    random.seed(cfg.seed)
    dataset_type = getattr(cfg.data, 'dataset_type', 'bimanual')
    if dataset_type == 'bkl':
        train_dataset = BKL_Dataset(
            features=cfg.data.features,
            demo_root=cfg.data.demo_root,
            demo_dirs=cfg.data.demo_dirs,
            inmem=cfg.data.inmem,
            start_ind=cfg.data.offset,
            num_demos=cfg.data.num_train,
            num_steps=cfg.actor.num_steps + cfg.actor.num_pred,
            num_pred=cfg.actor.num_pred,
            look_ahead=cfg.actor.look_ahead,
            im_size=cfg.data.im_size,
            cams=cfg.data.cams,
            noisy_skip=cfg.data.noisy_skip,
            frame_skip=cfg.data.frame_skip,
            joint_noise_std=cfg.data.joint_noise_std,
            joint_noise_std_scale=cfg.data.joint_noise_std_scale,
            feats_noise_std=cfg.data.feats_noise_std,
            history_repeating=cfg.data.history_repeating,
            img_sample_num=cfg.data.img_sample_num,
            prompt_text=getattr(cfg.data, 'prompt_text', 'pour the sugar'),
            prompt_embedding=getattr(cfg.data, 'prompt_embedding', None),
            prompt_embedding_path=getattr(cfg.data, 'prompt_embedding_path', None),

            action_stats_path=getattr(cfg.data, 'action_stats_path', None),
            noise_stats_path=getattr(cfg.data, 'noise_stats_path', None),
            side=getattr(cfg.data, 'side', 'both'),
        )
    else:
        train_dataset = Bimanual_Dataset(
            features=cfg.data.features,
            demo_root=cfg.data.demo_root,
            demo_dirs=cfg.data.demo_dirs,
            inmem=cfg.data.inmem,
            start_ind=cfg.data.offset,
            num_demos=cfg.data.num_train,
            num_steps=cfg.actor.num_steps + cfg.actor.num_pred,
            num_pred=cfg.actor.num_pred,
            look_ahead=cfg.actor.look_ahead,
            im_size=cfg.data.im_size,
            cams=cfg.data.cams,
            noisy_skip=cfg.data.noisy_skip,
            frame_skip=cfg.data.frame_skip,
            default_pos_left_arm=cfg.data.default_pos_left_arm,
            default_pos_right_arm=cfg.data.default_pos_right_arm,
            joint_noise_mean=cfg.data.joint_noise_mean,
            joint_noise_std=cfg.data.joint_noise_std,
            joint_noise_std_scale=cfg.data.joint_noise_std_scale,
            feats_noise_std=cfg.data.feats_noise_std,
            data_filter=cfg.data.data_filter,
            history_repeating=cfg.data.history_repeating,
            img_sample_num=cfg.data.img_sample_num,
            use_all_features=False,
            action_data_ratio=cfg.data.action_data_ratio,
            use_touch=cfg.data.use_touch,
            skip_failure=cfg.data.skip_failure,
        )
            
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=(cfg.train.mb_size // cfg.num_gpus),
        shuffle=(cfg.num_gpus == 1),
        sampler=DistributedSampler(train_dataset, shuffle=True) if cfg.num_gpus > 1 else None,
        num_workers=8, pin_memory=True, drop_last=True, persistent_workers=True
    )
    train_meter = TrainMeter(cfg.train.num_ep, len(train_loader))

    # Construct test dataset/loader
    if cfg.data.num_test > 0:
        random.seed(cfg.seed)
        if dataset_type == 'bkl':
            test_dataset = BKL_Dataset(
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
                joint_noise_std=0.0,
                joint_noise_std_scale=cfg.data.joint_noise_std_scale,
                feats_noise_std=0.0,
                history_repeating=cfg.data.history_repeating,
                img_sample_num=cfg.data.img_sample_num,
                prompt_text=getattr(cfg.data, 'prompt_text', 'pour the sugar'),
                prompt_embedding=getattr(cfg.data, 'prompt_embedding', None),
                prompt_embedding_path=getattr(cfg.data, 'prompt_embedding_path', None),
    
                action_stats_path=getattr(cfg.data, 'action_stats_path', None),
                side=getattr(cfg.data, 'side', 'both'),
            )
        else:
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
                feats_noise_std=0.0,
                data_filter=cfg.data.data_filter,
                history_repeating=cfg.data.history_repeating,
                img_sample_num=cfg.data.img_sample_num,
                use_all_features=False,
                action_data_ratio=cfg.data.action_data_ratio,
                use_touch=cfg.data.use_touch,
                skip_failure=cfg.data.skip_failure,
            )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=(cfg.train.mb_size // cfg.num_gpus),
            shuffle=False,
            num_workers=4, pin_memory=True, drop_last=False
        )
        test_meter = TestMeter(cfg.train.num_ep, len(test_loader))


    # Construct the model
    assert cfg.actor.type in ["transformer", "transformer_concat", "transformer_attnpool",
                              "transformer_concat_attnpool", "transformer_concat_meanpool", "transformer_temporalconcat",
                              "transformer_concat_attnpool_diffusion"]
    prompt_output_dim = [cfg.actor.prompt_dim]
    img_output_dim = [cfg.actor.im_dim for _ in range(len(cfg.data.cams))]
    state_dim = getattr(cfg.data, 'state_dim', 84 if getattr(cfg.data, 'use_touch', False) else 24)
    action_dim = getattr(cfg.data, 'action_dim', 24)
    state_output_dim = [state_dim]
    action_output_dim = [action_dim]
    output_dims = prompt_output_dim + img_output_dim + state_output_dim + action_output_dim
    if cfg.actor.type == "transformer_concat":
        model = ActorTransformerConcat_Bimanual(
            context_length=cfg.actor.num_steps,
            obs_shape=cfg.actor.obs_dim,
            output_dims=output_dims,
            num_pred=cfg.actor.num_pred,
            policy_cfg=cfg.transformer_concat,
            normalize_input=cfg.transformer_concat.normalize_input,
            normalize_bn_cls=nn.BatchNorm1d,
        )
    elif cfg.actor.type == "transformer_concat_attnpool":
        model = ActorTransformerConcatAttnPooling_Bimanual(
            context_length=cfg.actor.num_steps,
            obs_shape=cfg.actor.obs_dim,
            output_dims=output_dims,
            num_pred=cfg.actor.num_pred,
            policy_cfg=cfg.transformer_concat,
            normalize_input=cfg.transformer_concat.normalize_input,
            normalize_bn_cls=nn.BatchNorm1d,
            num_cam=len(cfg.data.cams),
            im_dim=cfg.actor.im_dim,
            prompt_dim=cfg.actor.prompt_dim,
        )
    elif cfg.actor.type == "transformer_concat_attnpool_diffusion":
        model = ActorTransformerConcatAttnPoolingDiffusion_Bimanual(
            context_length=cfg.actor.num_steps,
            obs_shape=cfg.actor.obs_dim,
            output_dims=output_dims,
            num_pred=cfg.actor.num_pred,
            policy_cfg=cfg.transformer_concat,
            normalize_input=cfg.transformer_concat.normalize_input,
            normalize_bn_cls=nn.BatchNorm1d,
            num_cam=len(cfg.data.cams),
            im_dim=cfg.actor.im_dim,
            prompt_dim=cfg.actor.prompt_dim,
            num_diffusion_steps=cfg.diffusion.num_diffusion_steps,
        )
    elif cfg.actor.type == "transformer_concat_meanpool":
        model = ActorTransformerConcatMeanPooling_Bimanual(
            context_length=cfg.actor.num_steps,
            obs_shape=cfg.actor.obs_dim,
            output_dims=output_dims,
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
    if cfg.train.weights:
        state_dict = torch.load(cfg.train.weights, map_location="cpu")["model_state"]
        mks, uks = model.load_state_dict(state_dict, strict=False)
        print("loaded checkpoint from: {}".format(cfg.train.weights))
        print("missing key: {}".format(mks))
        print("unexpected keys: {}".format(uks))

    # Transfer the model to the gpu
    model = model.cuda()

    # Construct model replicas
    if cfg.num_gpus > 1:
        cur_device = torch.cuda.current_device()
        model = DDP(module=model, device_ids=[cur_device], output_device=cur_device)

    # Construct the optimizer
    cfg.train.lr = cfg.train.lr * cfg.train.mb_size / 4096
    optimizer = torch.optim.AdamW(
        get_optimizer_groups(model, default_wd=cfg.train.wd),
        lr=cfg.train.lr,
        weight_decay=cfg.train.wd
    )

    if cfg.test_only:
        assert cfg.data.num_test > 0, "test-only mode needs data.num_test > 0"
        test_epoch(cfg, test_loader, model, test_meter, 0)
        return

    # Perform training
    for cur_epoch in range(cfg.train.num_ep):
        # Shuffle the data
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(cur_epoch)
        # Perform a training epoch
        train_epoch(cfg, train_loader, model, optimizer, train_meter, cur_epoch)
        # Evaluate the model
        if (cur_epoch + 1) % cfg.test.freq == 0 or (cur_epoch + 1) == cfg.train.num_ep:
            if cfg.data.num_test > 0:
                test_epoch(cfg, test_loader, model, test_meter, cur_epoch)
        # Save a checkpoint
        if (cur_epoch  + 1) % cfg.save_freq == 0 or (cur_epoch + 1) == cfg.train.num_ep:
            if cfg.num_gpus == 1 or dist.get_rank() == 0:
                save_checkpoint(cfg.logdir, model, cur_epoch + 1)

    print("Wrote results to: {}".format(cfg.logdir))

#!/usr/bin/env python3

"""Actor."""

from typing import Optional
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from timm.models.vision_transformer import Block as ViTEncoderBlock
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from mvp.models.policy import TransformerPolicy
from mvp.models.policy import DiffusionPolicyHead
from mvp.models.policy import get_1d_sincos_pos_embed as get_pos_embed
from mvp.utils.utils import masked_mse_loss


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


def temporal_dropout(x, drop_ratio, keep_last=True):
    if drop_ratio > 0.0:
        keep_ratio = 1.0 - drop_ratio
        mask = torch.empty([x.shape[0], x.shape[1], 1], dtype=x.dtype, device=x.device)
        mask.bernoulli_(keep_ratio)
        if keep_last:
            mask[:, -1] = 1
            keep_ratio = (1.0 + x.shape[1] * keep_ratio) / x.shape[1]
        x.div_(keep_ratio)
        x.mul_(mask)
    return x


class Attn_Pool(nn.Module):
    """ Attention pooling w/ latent query
    """
    def __init__(
            self,
            embed_dim: int = None,
            prompt_dim: int = None,
            num_prompt: int = 1,
            head_dim: int = 64,
            qkv_bias: bool = True,
            qk_norm: bool = False,
            norm_layer: Optional[nn.Module] = nn.LayerNorm,
            drop: float = 0.0,
    ):
        super().__init__()
        assert embed_dim % head_dim == 0
        self.num_heads = embed_dim // head_dim
        self.head_dim = head_dim
        self.scale = self.head_dim ** -0.5

        # self.prompt = nn.Parameter(torch.randn(1, 2, prompt_dim))
        self.q = nn.Linear(prompt_dim, embed_dim, bias=qkv_bias)
        self.kv = nn.Linear(embed_dim, embed_dim * 2, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.proj = nn.Linear(embed_dim * num_prompt, embed_dim)
        self.proj_drop = nn.Dropout(drop)

    def forward(self, im_emb, prompt, return_attention=False):
        b, T, N, C = im_emb.shape
        _, __, Nq, Cq = prompt.shape

        B = b * T
        im_emb = im_emb.reshape(b*T, N, C)
        prompt = prompt.reshape(b*T, Nq, Cq)
        # prompt = self.prompt.repeat(B, 1, 1)

        q = self.q(prompt).reshape(B, Nq, self.num_heads, self.head_dim).transpose(1, 2)

        kv = self.kv(im_emb).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)

        q, k = self.q_norm(q), self.k_norm(k)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        x = attn @ v

        if return_attention:
            att_each_head = []
            for head_ind in range(self.num_heads):
                att = attn[:, head_ind, 0, 1:].reshape(-1, 16, 16)
                att = F.interpolate(att.unsqueeze(1), (256, 256), mode='nearest').squeeze()  # B * 256 * 256
                # att = (att - att.flatten(1, 2).min(dim=-1).values[..., None, None]) / \
                #       (att.flatten(1, 2).max(dim=-1).values[..., None, None] - att.flatten(1, 2).min(dim=-1).values[..., None, None])
                att_each_head.append(att)

            return att_each_head


        x = x.transpose(1, 2).reshape(b, T, Nq*C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x



class ActorMLP(nn.Module):

    def __init__(self, cmd_dim, obs_dim, act_dim, model_cfg, num_steps):
        super(ActorMLP, self).__init__()
        assert num_steps == 1

        # Policy config
        actor_hidden_dim = model_cfg.ws
        emb_dim = actor_hidden_dim[0]

        # Input encoders
        self.cmd_proj = nn.Linear(cmd_dim, emb_dim // 2)
        self.obs_proj = nn.Linear(obs_dim, emb_dim // 2)

        # Policy
        activation = nn.SELU()
        actor_layers = []
        actor_layers.append(nn.Linear(emb_dim, actor_hidden_dim[0]))
        actor_layers.append(activation)
        for li in range(len(actor_hidden_dim)):
            if li == len(actor_hidden_dim) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dim[li], act_dim))
            else:
                actor_layers.append(
                    nn.Linear(actor_hidden_dim[li], actor_hidden_dim[li + 1])
                )
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Initialize the weights like in stable baselines
        actor_weights = [np.sqrt(2)] * len(actor_hidden_dim)
        actor_weights.append(0.01)
        assert model_cfg.init in ["orthogonal", "xavier_uniform"]
        self.init_weights(self.actor, actor_weights, model_cfg.init)

    @staticmethod
    def init_weights(sequential, scales, init_method):
        if init_method == "orthogonal":
            [
                torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
                for idx, module in enumerate(
                    mod for mod in sequential if isinstance(mod, nn.Linear)
                )
            ]
        elif init_method == "xavier_uniform":
            for module in sequential:
                if isinstance(module, nn.Linear):
                    torch.nn.init.xavier_uniform_(module.weight)
        else:
            raise NotImplementedError

    def forward(self, cmd, obs, act):
        # Squeeze the temporal dim
        cmd = cmd.squeeze(1)
        obs = obs.squeeze(1)
        # Compute cmd/obs embedding: (B, hidden_dim / 2)
        cmd_emb = self.cmd_proj(cmd)
        obs_emb = self.obs_proj(obs)
        # Joint embedding: (B, hidden_dim)
        cat_emb = torch.cat([cmd_emb, obs_emb], dim=1)
        # Compute the action
        actions = self.actor(cat_emb)
        return actions

    @torch.no_grad()
    def act_inference(self, cmd, obs, act):
        return self.forward(cmd, obs, act)


class ActorTransformerConcat(nn.Module):

    def __init__(
        self,
        context_length,
        obs_shape,
        actions_shape,
        num_pred,
        policy_cfg,
        normalize_input=False,
        normalize_bn_cls=None,
    ):
        super(ActorTransformerConcat, self).__init__()
        self.normalize_input = normalize_input
        self.causal = policy_cfg.causal
        self.num_pred = num_pred
        # Actor
        self.actor = TransformerPolicy(
            context_length=context_length,
            input_dim=obs_shape,
            output_dim=actions_shape * num_pred,
            in_proj_hidden_sizes=policy_cfg["in_proj_hidden_sizes"],
            embed_dim=policy_cfg["embed_dim"],
            num_blocks=policy_cfg["num_blocks"],
            num_heads=policy_cfg["num_heads"],
            mlp_ratio=policy_cfg["mlp_ratio"],
            attn_dropout=policy_cfg["attn_dropout"],
            mlp_dropout=policy_cfg["mlp_dropout"],
            last_layer_norm=policy_cfg["last_layer_norm"],
            head_hidden_sizes=policy_cfg["head_hidden_sizes"]
        )
        # Normalization
        if normalize_input:
            self.obs_normalizer = normalize_bn_cls(obs_shape, affine=False, momentum=None)
        self.mask_token = nn.Parameter(torch.zeros(obs_shape))

    def forward(self, observations, mod_mask=None, masks=None):
        L, B, C = observations.shape
        if mod_mask is not None:
            mask_token = self.mask_token[None, None].repeat(L, B, 1)
            observations = observations * mod_mask + mask_token * (1 - mod_mask)
        if self.normalize_input:
            observations = self.normalize_input_trajectory(observations)
        actions_mean = self.actor(observations, masks)
        return actions_mean

    @torch.no_grad()
    def act_inference(self, observations, state_buff, masks=None, mod_mask=None):
        return self.forward(observations, mod_mask, masks)

    def register_input_stats(self, observations):
        # Here we only update the running mean and variance of the observations
        _ = self.obs_normalizer(observations.view(-1, *observations.shape[2:])).view_as(observations)

    def normalize_input_trajectory(self, observations):
        # Here the implementation is a bit twisted.
        # We want to normalize all observations and states, but we need to avoid
        # recomputing the normalization statistics on invidual steps.
        # Therefore, we use eval mode for the normalization all on steps,
        # but the train mode for the latest step in each trajectory.
        normalizer_in_train_mode = self.obs_normalizer.training
        if normalizer_in_train_mode:  # We need to register the stats
            self.register_input_stats(observations[-1:])  # The latest step in each trajectory
        self.obs_normalizer.eval()
        observations = self.obs_normalizer(observations.reshape(-1, *observations.shape[2:])).view_as(observations)
        # Switch back to train mode if needed
        if normalizer_in_train_mode:
            self.obs_normalizer.train()
        return observations



class ActorTransformerConcat_Bimanual(ActorTransformerConcat):

    def __init__(
        self,
        output_dims,
        **kwargs,
    ):
        super(ActorTransformerConcat_Bimanual, self).__init__(actions_shape=sum(output_dims), **kwargs)
        self.output_dims = output_dims

    def forward(self, obs_each_mod=None, mask_each_mod=None, attn_mask=None, **kwargs):
        observations = torch.cat(obs_each_mod, dim=-1).transpose(0, 1)

        L, B, C = observations.shape
        if mask_each_mod is not None:
            mod_mask = torch.cat(mask_each_mod, dim=-1).transpose(0, 1)
            observations = observations * mod_mask + self.mask_token * (1 - mod_mask)
        if self.normalize_input:
            observations = self.normalize_input_trajectory(observations)

        preds = self.actor(observations, attn_mask)   # B * (num_pred * C)

        preds = preds.view(preds.shape[0], self.num_pred, -1)
        preds_each_mod = torch.split(preds, self.output_dims, dim=-1)  # prompt, ims, state, action

        return list(preds_each_mod)


class ActorTransformerConcatAttnPooling_Bimanual(ActorTransformerConcat):

    def __init__(
        self,
        output_dims,
        num_cam,
        im_dim,
        prompt_dim,
        **kwargs,
    ):
        super(ActorTransformerConcatAttnPooling_Bimanual, self).__init__(actions_shape=sum(output_dims), **kwargs)

        self.output_dims = output_dims
        self.attn_pool = nn.ModuleList([Attn_Pool(im_dim, prompt_dim) for _ in range(num_cam)])

        # self.contrastive_projector = nn.Linear(im_dim, prompt_dim, bias=False)

    def forward(self, obs_each_mod=None, mask_each_mod=None, attn_mask=None, im_features=None, prompts=None, cam_ids=None, attn_pool_only=False, **kwargs):
        # only attn-pool the image features
        if attn_pool_only:
            assert len(im_features) == len(cam_ids)
            prompts = prompts.unsqueeze(-2)  # B, T, Nq, C
            pooled_features = [self.attn_pool[i](im_features_c, prompts) for i, im_features_c in zip(cam_ids, im_features)]
            return pooled_features

        observations = torch.cat(obs_each_mod, dim=-1).transpose(0, 1)

        L, B, C = observations.shape
        if mask_each_mod is not None:
            mod_mask = torch.cat(mask_each_mod, dim=-1).transpose(0, 1)
            observations = observations * mod_mask + self.mask_token * (1 - mod_mask)
        if self.normalize_input:
            observations = self.normalize_input_trajectory(observations)

        preds = self.actor(observations, attn_mask)  # B * (num_pred * C)

        preds = preds.view(preds.shape[0], self.num_pred, -1)
        preds_each_mod = torch.split(preds, self.output_dims, dim=-1)  # prompt, ims, state, action
        return list(preds_each_mod)

    def vis_att(self, im_features=None, prompts=None, cam_ids=None):
        # only attn-pool the image features
        assert len(im_features) == len(cam_ids)
        prompts = prompts.unsqueeze(-2)  # B, T, Nq, C
        attentions = [self.attn_pool[i](im_features_c, prompts, return_attention=True) for i, im_features_c in
                           zip(cam_ids, im_features)]  # each item is a list of attention maps of different heads
        return attentions


class ActorTransformerConcatAttnPoolingDiffusion_Bimanual(ActorTransformerConcat):

    def __init__(
        self,
        output_dims,
        num_cam,
        im_dim,
        prompt_dim,
        num_diffusion_steps,
        **kwargs,
    ):
        super(ActorTransformerConcatAttnPoolingDiffusion_Bimanual, self).__init__(actions_shape=sum(output_dims), **kwargs)

        self.output_dims = output_dims
        self.attn_pool = nn.ModuleList([Attn_Pool(im_dim, prompt_dim) for _ in range(num_cam)])

        self.num_diffusion_steps = num_diffusion_steps
        self.action_diffusion_head = DiffusionPolicyHead(context_length=kwargs["num_pred"], input_dim=24,
                                                         input_condition_dim=24, output_dim=24,
                                                         num_diffusion_steps=num_diffusion_steps)
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=num_diffusion_steps,
            # the choise of beta schedule has big impact on performance
            # we found squared cosine works the best
            beta_schedule='squaredcos_cap_v2',
            clip_sample=False,
            # our network predicts noise (instead of denoised action)
            prediction_type='epsilon'
        )

    def forward(self, obs_each_mod=None, mask_each_mod=None, attn_mask=None, im_features=None, prompts=None, cam_ids=None, attn_pool_only=False, **kwargs):
        # only attn-pool the image features
        if attn_pool_only:
            assert len(im_features) == len(cam_ids)
            prompts = prompts.unsqueeze(-2)  # B, T, Nq, C
            pooled_features = [self.attn_pool[i](im_features_c, prompts) for i, im_features_c in zip(cam_ids, im_features)]
            return pooled_features

        observations = torch.cat(obs_each_mod, dim=-1).transpose(0, 1)

        L, B, C = observations.shape
        if mask_each_mod is not None:
            mod_mask = torch.cat(mask_each_mod, dim=-1).transpose(0, 1)
            observations = observations * mod_mask + self.mask_token * (1 - mod_mask)
        if self.normalize_input:
            observations = self.normalize_input_trajectory(observations)

        preds = self.actor(observations, attn_mask)  # B * (num_pred * C)

        preds = preds.view(preds.shape[0], self.num_pred, -1)
        preds_each_mod = torch.split(preds, self.output_dims, dim=-1)  # prompt, ims, state, action
        return list(preds_each_mod)

    def compute_diffusion_loss(self, latent, gt_actions, loss_mask):
        b = gt_actions.shape[0]

        # sample noises that will be added to the actions
        noise = torch.randn_like(gt_actions)

        # sample a time step for each instance
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (b,), device=gt_actions.device
        ).long()

        # Add noise to the clean gt actions according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_actions = self.noise_scheduler.add_noise(
            gt_actions, noise, timesteps)

        # Predict the noise residual
        pred = self.action_diffusion_head(noisy_actions, timesteps, latent)

        # Calculate the loss
        loss = masked_mse_loss(pred, noise, loss_mask)

        return loss

    def inference_diffusion(self, latent):
        # initial actions are gaussian noise
        B, N, _ = latent.shape
        actions = torch.randn(B, N, 24, device=latent.device)

        # set step values
        self.noise_scheduler.set_timesteps(self.num_diffusion_steps)

        # denoise
        for t in self.noise_scheduler.timesteps:
            # 1. predict model output
            model_output = self.action_diffusion_head(actions, torch.LongTensor([t], device=latent.device), latent)
            # 2. compute previous image: x_t -> x_t-1
            actions = self.noise_scheduler.step(
                model_output, t, actions,
            ).prev_sample

        return actions

    def vis_att(self, im_features=None, prompts=None, cam_ids=None):
        # only attn-pool the image features
        assert len(im_features) == len(cam_ids)
        prompts = prompts.unsqueeze(-2)  # B, T, Nq, C
        attentions = [self.attn_pool[i](im_features_c, prompts, return_attention=True) for i, im_features_c in
                           zip(cam_ids, im_features)]  # each item is a list of attention maps of different heads
        return attentions


class ActorTransformerConcatMeanPooling_Bimanual(ActorTransformerConcat):

    def __init__(
        self,
        output_dims,
        num_cam,
        im_dim,
        prompt_dim,
        **kwargs,
    ):
        super(ActorTransformerConcatMeanPooling_Bimanual, self).__init__(actions_shape=sum(output_dims), **kwargs)

        self.output_dims = output_dims

        # self.contrastive_projector = nn.Linear(im_dim, prompt_dim, bias=False)

    def forward(self, obs_each_mod=None, mask_each_mod=None, attn_mask=None, im_features=None, prompts=None, cam_ids=None, attn_pool_only=False, **kwargs):
        # only attn-pool the image features
        if attn_pool_only:
            assert len(im_features) == len(cam_ids)
            pooled_features = [im_features_c.mean(dim=-2) for i, im_features_c in zip(cam_ids, im_features)]
            return pooled_features

        observations = torch.cat(obs_each_mod, dim=-1).transpose(0, 1)

        L, B, C = observations.shape
        if mask_each_mod is not None:
            mod_mask = torch.cat(mask_each_mod, dim=-1).transpose(0, 1)
            observations = observations * mod_mask + self.mask_token * (1 - mod_mask)
        if self.normalize_input:
            observations = self.normalize_input_trajectory(observations)

        preds = self.actor(observations, attn_mask)  # B * (num_pred * C)

        preds = preds.view(preds.shape[0], self.num_pred, -1)
        preds_each_mod = torch.split(preds, self.output_dims, dim=-1)  # prompt, ims, state, action
        return list(preds_each_mod)

    def vis_att(self, im_features=None, prompts=None, cam_ids=None):
        # only attn-pool the image features
        assert len(im_features) == len(cam_ids)
        prompts = prompts.unsqueeze(-2)  # B, T, Nq, C
        attentions = [self.attn_pool[i](im_features_c, prompts, return_attention=True) for i, im_features_c in
                           zip(cam_ids, im_features)]  # each item is a list of attention maps of different heads
        return attentions

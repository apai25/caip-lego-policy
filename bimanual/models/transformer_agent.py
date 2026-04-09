"""Transformer agent."""
import math
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# from mvp.vision_model.vision_encoder import Encoder
from mvp.bimanual_bc.actor import ActorTransformerConcat_Bimanual, ActorTransformerConcatAttnPooling_Bimanual, \
    ActorTransformerConcatMeanPooling_Bimanual, ActorTransformerConcatAttnPoolingDiffusion_Bimanual
from mvp.bimanual_bc.dataset import compute_state, process_image, process_image_no_normalize

# from open_clip import create_model_and_transforms, get_tokenizer
from visual_feature_selector_yoloworld_sam2 import VisualFeatureSelector


def format_actions(actions, default_pos_left_arm, default_pos_right_arm):
    return {
        "left_arm_cmd": actions[:6] + np.array(default_pos_left_arm),
        "right_arm_cmd": actions[6:12] + np.array(default_pos_right_arm),
        "left_fingers_cmd": actions[12:18],
        "right_fingers_cmd": actions[18:24],
    }

class TransformerAgent:

    def __init__(self, cfg):

        self.cfg = cfg
        self.default_pos_left_arm = cfg.data.default_pos_left_arm
        self.default_pos_right_arm = cfg.data.default_pos_right_arm

        # creating the vision model
        if cfg.clip_attn_pool:
            self.clip_model, _, self.im_preprocess = create_model_and_transforms(cfg.clip_model_name, pretrained=cfg.clip_model_dir)
            self.text_tokenizer = get_tokenizer(cfg.clip_model_name)
            self.clip_model.cuda()
            self.clip_model.eval()

            self.clip_prompt_text = cfg.clip_prompt
            self.clip_prompt = self.text_tokenizer([self.clip_prompt_text]).cuda()
        elif cfg.visual_feature_selector:
            self.visual_feature_selector_0 = VisualFeatureSelector(visual_feature_selector_path='/home/ilija/code/mvp_generalize_v2/YoloWorld-SAM2', prompt_text = cfg.visual_feature_selector_prompt)
            self.visual_feature_selector_0.reset()
            self.visual_feature_selector_1 = VisualFeatureSelector(
                visual_feature_selector_path='/home/ilija/code/mvp_generalize_v2/YoloWorld-SAM2',
                prompt_text=cfg.visual_feature_selector_prompt)
            self.visual_feature_selector_1.reset()
            self.visual_feature_selector_2 = VisualFeatureSelector(
                visual_feature_selector_path='/home/ilija/code/mvp_generalize_v2/YoloWorld-SAM2',
                prompt_text=cfg.visual_feature_selector_prompt)
            self.visual_feature_selector_2.reset()

        else:
            self.im_enc = Encoder(
                cfg.encoder.name,
                cfg.encoder.pretrain_dir,
                cfg.encoder.freeze,
            )
            self.im_enc = self.im_enc.cuda()
            self.im_enc.eval()

        # choose which language model to use
        language_model = getattr(cfg, "language_model", "imagebind")
        if language_model == 'imagebind':
            from mvp.language_model.imagebind_language_model import get_text_embedding
        elif language_model == 't5':
            from mvp.language_model.sentence_t5 import get_text_embedding
        elif language_model == 'clip':
            from mvp.language_model.clip_language_model import get_text_embedding
        else:
            raise NotImplementedError

        self.get_text_embedding = get_text_embedding

        # creating the policy model
        assert cfg.actor.type in ["transformer_concat", "transformer_concat_attnpool",
                                  "transformer_concat_attnpool_diffusion", "transformer_concat_meanpool"]
        prompt_output_dim = [cfg.actor.prompt_dim ]
        img_output_dim = [cfg.actor.im_dim for _ in range(len(cfg.data.cams))]
        state_output_dim = [84 if cfg.data.use_touch else 24]
        action_output_dim = [24]
        output_dims = prompt_output_dim + img_output_dim + state_output_dim + action_output_dim
        if cfg.actor.type == "transformer_concat":
            self.model = ActorTransformerConcat_Bimanual(
                context_length=cfg.actor.num_steps,
                obs_shape=cfg.actor.obs_dim,
                output_dims=output_dims,
                num_pred=cfg.actor.num_pred,
                policy_cfg=cfg.transformer_concat,
                normalize_input=cfg.transformer_concat.normalize_input,
                normalize_bn_cls=nn.BatchNorm1d,
            )
        elif cfg.actor.type == "transformer_concat_attnpool":
            self.model = ActorTransformerConcatAttnPooling_Bimanual(
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
            self.model = ActorTransformerConcatAttnPoolingDiffusion_Bimanual(
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
            self.model = ActorTransformerConcatMeanPooling_Bimanual(
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

        # Load a checkpoint
        if cfg.test.weights:
            state_dict = torch.load(cfg.test.weights, map_location="cpu")["model_state"]
            mks, uks = self.model.load_state_dict(state_dict, strict=False)
            print("loaded checkpoint from: {}".format(cfg.test.weights))
            print("missing key: {}".format(mks))
            print("unexpected keys: {}".format(uks))

        self.model = self.model.cuda()
        self.model.eval()

        self.prompt = self.get_text_embedding(cfg.actor.prompt).float().cuda()

        self.state_dim = 84 if cfg.data.use_touch else 24
        self.prompt_dim = cfg.actor.prompt_dim
        self.process_img_every = math.ceil(cfg.actor.num_steps / cfg.data.img_sample_num)

    def set_prompt(self, prompt):
        self.prompt = self.get_text_embedding(prompt).float().cuda()

    def get_image_embedding(self, im, mode=None, prompt=None, cam_id=None, step = 0):
        if self.cfg.clip_attn_pool:
            im = Image.fromarray(im)
            im = self.im_preprocess(im)
            im = torch.tensor(im).cuda().unsqueeze(0)
            im_emb = self.clip_model.encode_image_with_text(im, self.clip_prompt).squeeze(0)
        elif self.cfg.visual_feature_selector:
            tmp_path = '/home/ilija/tmp/2024-9-30/{}/{}.png'.format(str(cam_id[0]), str(step))
            if cam_id == [0]:
                im_emb = self.visual_feature_selector_0.inference_frame(img_path=tmp_path, raw_image_array=im).squeeze(0)
            elif cam_id == [1]:
                im_emb = self.visual_feature_selector_1.inference_frame(img_path=tmp_path, raw_image_array=im).squeeze(0)
            else:
                im_emb = self.visual_feature_selector_2.inference_frame(img_path=tmp_path, raw_image_array=im).squeeze(0)
        else:
            im = process_image(im, self.cfg.data.im_size)
            im_t = torch.tensor(im).cuda().unsqueeze(0)
            im_emb = self.im_enc(im_t, mode=mode).unsqueeze(1)
            if 'attnpool' in self.cfg.actor.type or 'meanpool' in self.cfg.actor.type:
                im_emb = self.model(im_features=[im_emb],
                                    prompts=prompt,
                                    cam_ids=cam_id,
                                    attn_pool_only=True)[0].squeeze()
            elif im_emb.dim() == 4:
                im_emb = im_emb[:, :, 1:]
                B, T, N, C = im_emb.shape
                im_emb = im_emb.reshape(B*T, int(N**0.5), int(N**0.5), C)  # (B*T, H, W, C)
                im_emb = F.avg_pool2d(im_emb.permute(0, 3, 1, 2), kernel_size=self.cfg.data.img_downsample).permute(0, 2, 3, 1)  # (B*T, H/2, W/2, C)
                im_emb = im_emb.reshape(B, T, N // (self.cfg.data.img_downsample ** 2), C).squeeze()

        return im_emb

    def reset_buffers(self, obs = None):
        """
        Reset buffers
        """
        if 'all' in self.cfg.actor.feature_mode and 'attnpool' not in self.cfg.actor.type and 'meanpool' not in self.cfg.actor.type and not self.cfg.clip_attn_pool and not self.cfg.visual_feature_selector:
            self.im_buffs = [torch.zeros((1, self.cfg.actor.num_steps, 256 // (self.cfg.data.img_downsample ** 2), self.cfg.encoder.emb_dim)).float().cuda() for _ in self.cfg.data.cams]
        else:
            self.im_buffs = [torch.zeros((1, self.cfg.actor.num_steps, self.cfg.encoder.emb_dim)).float().cuda() for _ in self.cfg.data.cams]
        self.states_buff = torch.zeros((1, self.cfg.actor.num_steps, self.state_dim)).float().cuda()
        self.prompts_buff = torch.zeros((1, self.cfg.actor.num_steps, self.prompt_dim)).float().cuda()
        self.mod_mask = np.zeros((1, self.cfg.actor.num_steps, self.cfg.actor.obs_dim))
        if obs:
            for i, cam in enumerate(self.cfg.data.cams):
                im = obs[f"rgb_{cam}"]
                im_emb = self.get_image_embedding(im, mode=self.cfg.actor.feature_mode, prompt=self.prompts_buff[:, -1:], cam_id=[i])
                self.im_buffs[i][:, :].copy_(im_emb)


            state = compute_state(obs, self.default_pos_left_arm, self.default_pos_right_arm, use_touch=self.cfg.data.use_touch)
            state_t = torch.tensor(state).cuda().unsqueeze(0)
            self.states_buff[:, :].copy_(state_t)

            self.mod_mask[:, :, -self.state_dim:] = 1  # all states are visible
            self.mod_mask[:, :, :self.cfg.actor.prompt_dim] = 1  # all prompts are visible
            self.mod_mask[:, ::-self.process_img_every, self.cfg.actor.prompt_dim:-self.state_dim] = 1  # every (process_img_every) images are visible
            self.mod_mask = torch.tensor(self.mod_mask).float().cuda()

        self.att_mask = torch.ones(1, self.cfg.actor.num_steps).cuda()

        if self.prompt is not None:
            self.prompts_buff[:, -1].copy_(self.prompt)
        else:
            raise NotImplementedError

    @torch.no_grad()
    def act(self, obs, process_img=True, step = 0):

        # update img buffer
        if process_img:
            for i, cam in enumerate(self.cfg.data.cams):
                im = obs[f"rgb_{cam}"]
                im_emb = self.get_image_embedding(im, mode=self.cfg.actor.feature_mode, prompt=self.prompts_buff[:, -1:], cam_id=[i], step = step)
                self.im_buffs[i][:, -1].copy_(im_emb)
            self.mod_mask[:, -1, self.cfg.actor.prompt_dim:-self.state_dim] = 1
        else:
            self.mod_mask[:, -1, self.cfg.actor.prompt_dim:-self.state_dim] = 0

        state = compute_state(obs, self.default_pos_left_arm, self.default_pos_right_arm, use_touch=self.cfg.data.use_touch)
        state_t = torch.tensor(state).cuda().unsqueeze(0)
        self.states_buff[:, -1].copy_(state_t)
        self.mod_mask[:, -1, -self.state_dim:] = 1
        self.mod_mask[:, -1, :self.cfg.actor.prompt_dim] = 1

        self.att_mask[:, -1] = 0

        obs_each_mod = [self.prompts_buff] + self.im_buffs + [self.states_buff]
        mask_each_mod = list(self.mod_mask.split([self.cfg.actor.prompt_dim] + [self.cfg.actor.im_dim] * len(self.cfg.data.cams) + [self.state_dim], dim=-1))

        # construct attention mask
        att_mask = self.att_mask.unsqueeze(1).repeat(1, self.cfg.actor.num_steps, 1)  # B * L * L
        att_mask[:, range(self.cfg.actor.num_steps), range(self.cfg.actor.num_steps)] = 0  # diagonal elements must be 0
        B, L, L = att_mask.shape
        att_mask = att_mask.unsqueeze(1).repeat(1, self.cfg.transformer_concat.num_heads, 1, 1).view(B * self.cfg.transformer_concat.num_heads, L, L)
        att_mask = att_mask.bool()

        # construct causal mask
        if self.cfg.transformer_concat.causal:
            causal_mask = torch.triu(torch.ones(self.cfg.actor.num_steps, self.cfg.actor.num_steps), diagonal=1).cuda().bool()
            causal_mask = causal_mask.unsqueeze(0).repeat(att_mask.shape[0], 1, 1)
            att_mask = torch.logical_or(att_mask, causal_mask)

        # find the index of steps with images
        img_selected_ids = torch.all(self.mod_mask[0, :, self.cfg.actor.prompt_dim:-self.state_dim].bool(), dim=-1)
        img_selected_ids = img_selected_ids.nonzero(as_tuple=True)[0]
        img_selected_ids = img_selected_ids[None]

        preds_each_mod = self.model(obs_each_mod=obs_each_mod, mask_each_mod=mask_each_mod, attn_mask=att_mask, img_selected_ids=img_selected_ids)
        actions = preds_each_mod[-1]
        if "diffusion" in self.cfg.actor.type:
            actions = self.model.inference_diffusion(actions)

        actions = actions.reshape(self.cfg.actor.num_pred, -1)

        for i, _ in enumerate(self.cfg.data.cams):
            self.im_buffs[i][:, :-1] = self.im_buffs[i][:, 1:].clone()
        self.prompts_buff[:, :-1] = self.prompts_buff[:, 1:].clone()
        self.states_buff[:, :-1] = self.states_buff[:, 1:].clone()
        self.att_mask[:, :-1] = self.att_mask[:, 1:].clone()
        self.mod_mask[:, :-1] = self.mod_mask[:, 1:].clone()

        self.prompts_buff[:, -1] = self.prompt.clone()

        actions = actions.detach().cpu().numpy()
        return actions

#!/usr/bin/env python3

"""Run policy on bimanual robot."""
import math
import time
import warnings

import hydra
import numpy as np
import omegaconf
import redis
import torch
import torch.nn as nn
import torch.nn.functional as F
from dall_e import unmap_pixels
# from dall_e.decoder import Decoder
from dall_e.encoder import Encoder
from einops import rearrange
from huggingface_hub import hf_hub_download
from transformers import LlamaForCausalLM
from transformers.models.llama.configuration_llama import LlamaConfig

from bimanual.env.real import DualUR3ERealEnv
from mvp.bc.actor import (ActorTransformerConcat)
from mvp.bimanual_bc.dataset import compute_state, process_image
# from mvp.bimanual_bc.vision_encoder import Encoder
from mvp.language_model.imagebind_language_model import get_text_embedding


class Tokenizer(nn.Module):
    def __init__(self, model="dvae-8k", image_size=128, patch_size=8, path=None):
        super().__init__()

        self.model = model
        self.image_size = image_size
        self.patch_size = patch_size
        self.h_ = image_size // patch_size
        self.w_ = image_size // patch_size

        if(self.model=="dvae-8k"):
           # For faster load times, download these files locally and use the local paths instead.
            self.enc = Encoder()
            # self.dec = Decoder()

            self.enc.load_state_dict(torch.load(hf_hub_download(repo_id="brjathu/image_tokenizers", filename="dvae_8k_encoder.pt", token="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")))
            # self.dec.load_state_dict(torch.load(hf_hub_download(repo_id="brjathu/image_tokenizers", filename="dvae_8k_decoder.pt", token="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")))

#            _ = self.dec.blocks.group_1.upsample
#            self.dec.blocks.group_1.upsample = torch.nn.Upsample(scale_factor = _.scale_factor, mode= _.mode)
#            _ = self.dec.blocks.group_2.upsample
#            self.dec.blocks.group_2.upsample = torch.nn.Upsample(scale_factor = _.scale_factor, mode= _.mode)
#            _ = self.dec.blocks.group_3.upsample
#            self.dec.blocks.group_3.upsample = torch.nn.Upsample(scale_factor = _.scale_factor, mode= _.mode)

#            for n, p in self.dec.named_parameters():
#                p.requires_grad = False
            for n, p in self.enc.named_parameters():
                p.requires_grad = False

            self.enc.eval()
#            self.dec.eval()
            self.vocab_size = 8192

    def tokenize_img(self, img):
        # expects imagenet normalized images

        if("dvae" in self.model):
            device = img.device
            dtype = img.dtype
            logit_laplace_eps = 0.1

            imagenet_mean = torch.from_numpy(np.array([0.485, 0.456, 0.406])).to(device=device).to(dtype=dtype)
            imagenet_std = torch.from_numpy(np.array([0.229, 0.224, 0.225])).to(device=device).to(dtype=dtype)
            imgs_ = (img * imagenet_std[None, :, None, None]) + imagenet_mean[None, :, None, None]
            imgs_ = (1 - 2.0 * logit_laplace_eps) * imgs_ + logit_laplace_eps
            imgs_ = F.interpolate(imgs_, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
            with torch.no_grad():
                z_logits = self.enc(imgs_)
            z_ = torch.argmax(z_logits, axis=1)
            target = z_
            
        return target
    
    def reconstruct_img(self, tokens, h_tokens=16, w_tokens=16, patch=8):

        # target.shape: bs x 196
        bs = tokens.shape[0]

        if("dvae" in self.model):
            tokens = tokens.view(tokens.shape[0], h_tokens, w_tokens)
            # tokens to one hot
            z = F.one_hot(tokens, num_classes=self.enc.vocab_size).permute(0, 3, 1, 2).float()
            x_stats = self.dec(z).float()
            x_rec = unmap_pixels(torch.sigmoid(x_stats[:, :3]))
            img_recon = torch.clip(x_rec, 0, 1)
            img_recon = torch.einsum('nchw->nhwc', img_recon)
            img_recon = img_recon.detach().cpu().numpy()
            return img_recon

        return img_recon
    




def format_actions(actions, default_pos_left_arm, default_pos_right_arm):
    return {
        "left_arm_cmd": actions[:6] + np.array(default_pos_left_arm),
        "right_arm_cmd": actions[6:12] + np.array(default_pos_right_arm),
        "left_fingers_cmd": actions[12:18],
        "right_fingers_cmd": actions[18:24],
    }

def action_obs_diff(action, obs):
    # Get difference between action goal and measured obs after executing action
    action_keys = ['left_arm_cmd', 'right_arm_cmd', 'left_fingers_cmd', 'right_fingers_cmd']
    obs_keys = ['left_arm_joint_pos', 'right_arm_joint_pos', 'left_fingers_joint_pos', 'right_fingers_joint_pos']
    diff = {}
    for i in range(len(action_keys)):
        obs_val = np.array(obs[obs_keys[i]])
        action_val = np.array(action[action_keys[i]])
        if 'hand' in action_keys[i]:
            action_val = np.rad2deg(action_val)
        diff[action_keys[i]] = action_val - obs_val 
    return diff

class DummyAgent:

    def __init__(self, cfg):
        self.cfg = cfg

    def act(self, obs):
        return np.concatenate([obs["left_arm_joint_pos"] - np.array(self.cfg.data.default_pos_left_arm),
                               obs["right_arm_joint_pos"] - np.array(self.cfg.data.default_pos_right_arm),
                               np.deg2rad(obs["left_fingers_joint_pos"]),
                               np.deg2rad(obs["right_fingers_joint_pos"])])


class TransformerAgent:

    def __init__(self, cfg):

        self.cfg = cfg
        self.default_pos_left_arm = cfg.data.default_pos_left_arm
        self.default_pos_right_arm = cfg.data.default_pos_right_arm

        # image tokenizer
        hsize = 384
        self.proj_obs_in = nn.Linear(24, hsize)
        self.proj_obs_out = nn.Linear(hsize, 24)
        self.tokenizer = Tokenizer(model="dvae-8k", image_size=64, patch_size=8)
        self.tokenizer = self.tokenizer.cuda()
        self.proj_obs_in = self.proj_obs_in.cuda()
        self.proj_obs_out = self.proj_obs_out.cuda()
        

        # vgpt model
        hsize = 384
        isize = 768
        num_hidden_layers = 12
        num_attention_heads = 12
        vgpt_config = LlamaConfig()
        vgpt_config.intermediate_size = isize
        vgpt_config.hidden_size = hsize
        vgpt_config.max_position_embeddings = 4096+2
        vgpt_config.num_attention_heads = num_attention_heads
        vgpt_config.num_hidden_layers = num_hidden_layers
        vgpt_config.num_key_value_heads = num_attention_heads
        vgpt_config.vocab_size = self.tokenizer.vocab_size + 256
        vgpt_config.use_cache = True
        self.vgpt = LlamaForCausalLM(vgpt_config)
        self.vgpt = self.vgpt.cuda()

        self.process_img_every = math.ceil(cfg.actor.num_steps / cfg.data.img_sample_num)


    def tokenization(self, imgs_in, size=128):
        img = imgs_in
        device = img.device
        dtype = img.dtype
        
        tokens = self.tokenizer.tokenize_img(img)
        tokens = tokens + 10
        
        return tokens, device, dtype

    def reset_buffers(self, obs = None):
        """
        Reset buffers
        """
        self.im_buffs = [torch.zeros((1, self.cfg.actor.num_steps, 64)).float().cuda() for _ in self.cfg.data.cams]
        self.states_buff = torch.zeros((1, self.cfg.actor.num_steps, 24)).float().cuda()
        self.mod_mask = np.zeros((1, self.cfg.actor.num_steps, self.cfg.actor.obs_dim))
        if obs:
            for i, cam in enumerate(self.cfg.data.cams):
                im = obs[f"rgb_{cam}"]
                im = process_image(im, self.cfg.data.im_size)
                im_t = torch.tensor(im).cuda().unsqueeze(0)
                # tokeize the image and store it in the buffer
                tokens_, device, dtype = self.tokenization(im_t)
                tokens_ = tokens_.view(1, -1)
                self.im_buffs[i][:, :].copy_(tokens_)
            
            state = compute_state(obs, self.default_pos_left_arm, self.default_pos_right_arm)
            state_t = torch.tensor(state).cuda().unsqueeze(0)
            self.states_buff[:, :].copy_(state_t)

            self.mod_mask[:, :, -24:] = 1  # all states are visible
            self.mod_mask[:, :, :self.cfg.actor.prompt_dim] = 1  # all prompts are visible
            self.mod_mask[:, ::-self.process_img_every, self.cfg.actor.prompt_dim:-24] = 1  # every (process_img_every) images are visible
            self.mod_mask = torch.tensor(self.mod_mask).float().cuda()

        self.att_mask = torch.ones(1, self.cfg.actor.num_steps).cuda()

    @torch.no_grad()
    def act(self, obs, process_img=True):

        # update img buffer
        if process_img:
            for i, cam in enumerate(self.cfg.data.cams):
                im = obs[f"rgb_{cam}"]
                im = process_image(im, self.cfg.data.im_size)
                im_t = torch.tensor(im).cuda().unsqueeze(0)
                # tokeize the image and store it in the buffer
                tokens_, device, dtype = self.tokenization(im_t)
                tokens_ = tokens_.view(1, -1)
                self.im_buffs[i][:, -1].copy_(tokens_)
            self.mod_mask[:, -1, self.cfg.actor.prompt_dim:-24] = 1
        else:
            self.mod_mask[:, -1, self.cfg.actor.prompt_dim:-24] = 0

        state = compute_state(obs, self.default_pos_left_arm, self.default_pos_right_arm)
        state_t = torch.tensor(state).cuda().unsqueeze(0)
        self.states_buff[:, -1].copy_(state_t)
        self.mod_mask[:, -1, -24:] = 1
        self.mod_mask[:, -1, :self.cfg.actor.prompt_dim] = 1
        self.att_mask[:, -1] = 0




        tokens = torch.cat(self.im_buffs, dim=-1)
        tokens = rearrange(tokens, 'b t (n hw) -> b t n hw', n=3)
        tokens = tokens.to(torch.int)
        token_embs = self.vgpt.get_input_embeddings()(tokens)
        token_embs = rearrange(token_embs, 'b t n hw d -> b t (n hw) d')
        
        obs_emb = self.states_buff
        token_obs_emb = self.proj_obs_in(obs_emb)

        embeddings = torch.cat([token_embs, token_obs_emb[:, :, None, :]], dim=2) # torch.Size([1, 16, 193, 384])
        embeddings = rearrange(embeddings, 'b t n d -> b (t n) d')


        b1 = self.vgpt(inputs_embeds=embeddings, use_cache=False, output_hidden_states=True)
        # actions = self.model(pi_obs.transpose(0, 1), mod_mask=self.mod_mask.transpose(0, 1), masks=att_mask)
        # actions = actions.view(self.cfg.actor.num_pred, -1)[:, -24:]

        actions = b1['hidden_states'][-1]
        feats_per_step = rearrange(actions, "b (t n) d -> b t n d", t = 16)
        feats_per_step_act = feats_per_step[:, :, -1, :]
        feats_per_step_act_out = self.proj_obs_out(feats_per_step_act)
        actions = feats_per_step_act_out[:, -1]
        actions[:, :6] = self.states_buff[:, -1, :6]

        for i, _ in enumerate(self.cfg.data.cams):
            self.im_buffs[i][:, :-1] = self.im_buffs[i][:, 1:].clone()
        self.states_buff[:, :-1] = self.states_buff[:, 1:].clone()
        self.att_mask[:, :-1] = self.att_mask[:, 1:].clone()
        self.mod_mask[:, :-1] = self.mod_mask[:, 1:].clone()

        actions = actions.detach().cpu().numpy()
        return actions


@hydra.main(version_base=None, config_name="config", config_path="../../configs/bimanual_bc")
def run_robot(cfg: omegaconf.DictConfig):
    r = redis.Redis(host="localhost", port=6379, db=0)
    r.flushall()

    with DualUR3ERealEnv(window=False, show_cams=True) as env:
        control_freq = 15 
        num_steps = cfg.test.num_steps
        agent = TransformerAgent(cfg)

        env.init(random=False)
        obs = env.get_obs()

        
        agent.reset_buffers(obs)
        all_actions = np.zeros((num_steps, num_steps + cfg.actor.num_exec - 1, 24))

        assert cfg.actor.look_ahead == 0

        for step in range(num_steps):
            begin_t = time.time()

            # get action
            
            actions = agent.act(obs, process_img=(step % agent.process_img_every == 0))
            all_actions[step, step:step+cfg.actor.num_exec] = actions[:cfg.actor.num_exec]

            if cfg.actor.num_agg == -1:
                # no temporal aggregation
                action = all_actions[step - step % cfg.actor.num_exec, step]
            else:
                start_idx = max(step - cfg.actor.num_exec + 1, 0)
                end_idx = min(start_idx + cfg.actor.num_agg, step + 1)
                selected_actions = all_actions[start_idx: end_idx, step]
                weights = np.exp(-0.2 * np.array(list(range(selected_actions.shape[0]))))
                action = (selected_actions * weights[..., None]).sum(axis=0) / weights.sum()

            # Safety measure. If the movement in any joint is larger than 1 rad, then abort immediately.
            assert np.abs(action[:12] - compute_state(obs, cfg.data.default_pos_left_arm, cfg.data.default_pos_right_arm)[:12]).max() < 2.0, \
                    "Large joint movement detected! Abort now..."

            action = format_actions(action, cfg.data.default_pos_left_arm, cfg.data.default_pos_right_arm)

            # step env
            env.step(action)

            # sleep
            print(time.time() - begin_t)
            time.sleep(max(0, 1.0 / control_freq - (time.time() - begin_t)))
            obs = env.get_obs()



if __name__ == "__main__":
    run_robot()

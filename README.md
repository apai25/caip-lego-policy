# Generalizable Vision for Robotics Manipulation

## Environment setup

**This is the environment setup for training. Probably need additional setup for the vision models (e.g., open-world detector, SAM2, etc.)**

Create a conda environment:

```
conda create -n mvp-sim2real python=3.8
conda activate mvp-sim2real
```

Install [PyTorch](https://pytorch.org/get-started/locally/):

```
conda install pytorch==1.13.1 torchvision==0.14.1 pytorch-cuda=11.6 -c pytorch -c nvidia
```

Install IsaacGym (download [here](https://developer.nvidia.com/isaac-gym)):
(**I think we don't need this step now but need to check**)

```
cd /path/to/isaac-gym/python
pip install -e .
```

Clone this repo:

```
cd /path/to/code
git clone git@github.com:ir413/mvp-sim2real.git
```

Install Python dependencies:

```
cd /path/to/code/mvp-sim2real
pip install -r requirements.txt
```

Install this repo:

```
cd /path/to/code/mvp-sim2real
pip install -e .
```



## Data Preprocessing for Teleoperated Data

We use the supervised visual feature selector as an example.

For teleoperated data, assuming it's organized as 

```
- <demo_name>
   - <demo_1_name>
      - 0000.pkl
      - 0001.pkl
      ...
      - metadata.json
      - success.txt
   - <demo_2_name>
      ...
```

we need to extract the vision features of the target objects (e.g., yellow cube) using the supervised visual feature selector, label the language instructions (e.g., "pick right arm"), and extract the language embeddings of the instruction.

### Extract vision features

To extract vision features of demos saved under a directory `<root_folder>/<demo_folder>`, and save the demos with vision features to a target directory `<save_root_folder>/<demo_folder>`:

```
python tools/store_attn_pooled_features_visual_feature_selector.py --data-root <root_folder> --save-root <save_root_folder> --demo-name <demo_folder> --model-name <name_of_your_vision_model> --prompt <prompt_for_visual_selector> --keys left head right --fp16

## For example:
python tools/store_attn_pooled_features_visual_feature_selector.py --data-root /home/ilija/data/bimanual/ --save-root /home/bfshi/data/bimanual/features/features_attn_pooled_visual_feature_selector --demo-name pick-yellow-right_04-29-2024 --model-name dinov2-b --prompt yellow\ cube --keys left head right --fp16
```

### Label instructions

To add instruction labels to a directory of demos, run

```
python tools/label_instructions.py --path <demo_folder> --instruction <instruction>

## For example:
python tools/label_instructions.py --path /home/bfshi/data/bimanual/features/features_attn_pooled_visual_feature_selector/dinov2-b/pick-yellow-right_04-29-2024 --instruction pick\ right\ arm
```

### Extract language embedding of instructions

```
python tools/label_instruction_embedding.py --path <demo_folder> --language_model <which_language_model_to_use>

## For example:
python tools/label_instruction_embedding.py --path /home/bfshi/data/bimanual/features/features_attn_pooled_visual_feature_selector/dinov2-b/pick-yellow-right_04-29-2024 --language_model t5
```



## Training

Training on 120 pick-right demos
```
torchrun --nproc_per_node=4 --master_port=2343 tools/train_bimanual_bc.py num_gpus=4 data.demo_root=/home/bfshi/data/bimanual/features/features_attn_pooled_visual_feature_selector/dinov2-b data.demo_dirs=[pick-yellow-right_04-29-2024] data.num_train=10000 data.frame_skip=1 logdir=<output_dir> train.mb_size=2048 train.warmup_ep=30 train.num_ep=900 actor.type=transformer_concat actor.num_steps=16 actor.num_pred=16 actor.obs_dim=3096 actor.act_dim=3120 actor.im_dim=768 actor.prompt_dim=768 data.inmem=False data.img_sample_num=8 actor.state_loss_weight=0.03
```



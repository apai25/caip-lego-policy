# Bimanual UR3e robot

## Installation
Install dependencies in your python env.
```
pip install -r requirements.txt
```

## Collect Teleoperated Trajectories with Quest3 VR

### Setup Hardware
1. Turn on both UR3e arms and Ability hands. For both arms: switch to local mode, go to Move screen, move the arm a bit with the arrows, switch back to remote mode. 
2. Turn on Quest3 headset and connect headset to Linux machine via USB-C
3. In headset, click Bell Icon on menu bar. This will open a list of recent notifications. Click on the notification about USB connection to allow USB debugging. Wait a few seconds. 
4. After step 3, running `adb devices` in Linux shell should give the device ID. If it says unauthorized permissions, try unplugging/replugging usb connection and revisit step 3 to enable USB debugging.
5. Run Unity app by selecting `Folder Icon > My project app`
6. If you are using hand tracking, ensure hand tracking is enabled in the Quest3. If you are using MANUS mocap gloves, ensure hand tracking is disabled in the Quest3 and follow [MANUS setup guide](control/manus_sdk/README.md).
7. Run `python bimanual/scripts/view_cameras.py` to verify camera views (order is left, middle, right).
8. Walk next to robot and Recenter (very important because Quest3 hand rotations are global, not relative). To Recenter with controllers, hold the Meta logo on the right controller. To Recenter with hand tracking, pinch with index finger and thumb and hold. 

### Collect Demonstrations
1. Create a directory for your data in `~/data/`
2. Run data collection script, `python bimanual/scripts/collect_traj.py -—data-dir ~/data/pick-yellow-right_05-28-2024`
3. Wait for arms and hands to initialize.
4. Reset the physical environment for this demonstration. 
5. Use right-most pedal as deadman switch and record demo. With the deadman switch is not pressed, data collection will pause. 
6. Once finished, immediately press middle pedal for success OR left-most pedal for failure
7. Repeat for the next demo.

### Things to keep in mind
1. Try to match the random initialization with your hands at the beginning. Position your hands with the intended motion in mind (i.e. if arm initial pose starts very high up and you plan to reach down to the table to grab something, you want to position your hands very high up so you can execute the motion smoothly without having to release deadman switch to reposition hands).
2. Hands must always be in view of quest. If both hands overlap from the quests view, it might get confused. 
3. If the hand goes undetected (or has low confidence causing the hand tracking to fit incorrectly/noisly), the arm may retarget to some unexpected position (collisions won’t happen) or not move at all. Release the pedal to reposition your hands and continue.
4. If the hand sensor values stop updating, the script will automatically quit and the trajectory won’t be saved. You may want to power cycle (unplug hands, wait a few seconds, replug hands) or rest the hands unplugged for a bit before resuming. Ability hand triple beeping means the hands are overheating. 


## Visualize Teleoperated Trajectories

### Visualize Single Trajectory Locally through Rerun App

Run the following script to visualize a single trajectory using rerun-sdk. This will automatically open the Rerun GUI locally. Use this script to quickly visualize one trajectory.
```
python scripts/vis_traj.py --traj ~/data/demo_dataset/traj_name
```

### Visualize Many Trajectories Remotely through Browser

Run the following script to visualize a many trajectories using rerun-sdk. Use the localhost website outputted by the program to view data in the browser. Use different `--port` for visualizing different datasets (e.g. 9876, 9875).
```
python scripts/rerun_traj --data-dir ~/data/demo_dataset --num-traj 5 --port 9876
```

## Run real-world experiments

### Run policy

```
# One example script of running pick-yellow:
python scripts/run_robot.py --config-path /home/ilija/code/mvp-sim2real/output/240813_2148_reproduce test.weights=/home/ilija/code/mvp-sim2real/output/240813_2148_reproduce/model_ep0900.pt +encoder.emb_dim=768 +actor.feature_mode=all actor.num_exec=16 +test.num_steps=900 +actor.prompt=pick\ right\ arm +actor.num_agg=16 +clip_attn_pool=False +visual_feature_selector=True +visual_feature_selector_detector_path=/home/ilija/code/visual_feature_selector/assets/model_final.pth +visual_feature_selector_cfg=/home/ilija/code/visual_feature_selector/assets/configs/COCO-Detection/fatser_rcnn_R_101_FPN_bimanual.yaml +visual_feature_selector_prompt=red\ cube +language_model=t5
```

### Evaluation
Use `scripts/eval_pick.py`, `scripts/eval_sort.py`, and `scripts/eval_stack.py` for evaluating tasks. Use similar CLI arguments to `run_robot.py` script. The pick evaluation script will use the right hand to point to 12 initial cube locations. The stack and sort evaluation scripts will choose 12 random initial cube locations with fixed seed.



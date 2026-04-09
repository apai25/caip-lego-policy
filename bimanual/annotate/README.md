# Bimanual Trajectory Annotation

## Annotation Pipeline Using Webapp
1. Convert trajectories into head camera mp4 videos. Save mp4s per trajectory from dataset `/home/ilija/data/bimanual/<DATASET_NAME>/` into `/home/ilija/data/bimanual-annotate/mp4s/<DATASET_NAME>/`. 
```
python convert_mp4.py --traj-dir /home/ilija/data/bimanual/<DATASET_NAME>/ --video-dir /home/ilija/data/bimanual-annotate/mp4s/<DATASET_NAME>/
```

2. Launch the webapp to label trajectory segments with language annotations. The annotations will be found at `/home/ilija/data/bimanual-annotate/annotations/<DATASET_NAME>`. 
    1. Go into the `webapp` directory.
    ```
    cd webapp
    ```
    2. Install dependencies
    ```
    npm install cors express body-parser axios react-router-dom
    ```
    3. Build ReactJS frontend app. The static assets will be found in `build/`.
    ```
    npm run build
    ```
    4. Launch the NodeJS backend at port 3000.
    ```
    node server.js
    ```

3. Use the annotation jsons from `/home/ilija/data/bimanual-annotate/annotations/<DATASET_NAME>` to organize the training data into segmented trajectories in pkl format. The resulting segmented data will be saved at `/home/ilija/data/bimanual/labelled-<DATASET_NAME>`. 

Use `--combine` flag to merge contiguous annotated success trajectories into a single trajectory and save annotations per pkl file. This is useful for next prompt prediction. Data in combined format will be saved at `/home/ilija/data/bimanual/labelled-combined-<DATASET_NAME>`
```
python organize_segments.py --annotation-dir /home/ilija/data/bimanual-annotate/annotations/<DATASET_NAME> --traj-dir /home/ilija/data/bimanual/<DATASET_NAME>
```

## Annotation Pipeline Using Local App
1. Convert trajectories into head camera mp4 videos. This script will save trajectory data from `/path/to/trajs/<TRAJ_NAME>/` into `/path/to/mp4s/<TRAJ_NAME>.mp4`. 
```
python convert_mp4.py --traj-dir /path/to/trajs/ --video-dir /path/to/mp4s/
```
 
2. Launch the annotation GUI to segment trajectories and label language instructions per segment. The GUI will save annotations from `/path/to/mp4s/<TRAJ_NAME>.mp4` into `/path/to/annotations/<TRAJ_NAME>.json`. 
```
python annotate.py --video-dir /path/to/mp4s/ --annotation-dir /path/to/annotations/
```

3. Use the annotation jsons to organize the training data into segmented trajectories in pkl format. Provide paths to annotation jsons and original training data. The resulting segmented data will be at `/path/to/labelled-trajs/`
```
python organize_segments.py --annotation-dir /path/to/annotations/ --traj-dir /path/to/trajs/
```

## Annotation Guidelines
It is important to be consistent in annotating trajectories. The following are examples of annotations: 
* Pick up the purple cube with the right hand.
* Place the blue ball into the orange bowl with the right hand.
* Drop the orange ball into the yellow bin with the right hand.
* Hand off the red ball from the left to the right hand.
* Drag the bowl closer with the left hand.
* Push the blue ball leftward with the right hand.
* Toss the yellow ball into the blue bowl with the left hand.

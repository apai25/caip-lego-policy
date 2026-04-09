"""Use labels from annotation app to segment and save trajectories in pkl format."""

import os
import click
import json
import joblib
import traceback

from glob import glob
from multiprocessing import Pool
from tqdm import tqdm

def organize_seg(args):
    try:
        af, combine, traj_dir, labelled_data_dir = args
        with open(af, 'r') as f:
            annotations = json.load(f)
        if len(annotations) == 0:
            print('No annotations found in', af)
            return

        # Standardize keys
        for ann in annotations:
            if 'failed' in ann:
                ann['success'] = not ann.pop('failed')
            if 'text' in ann:
                ann['language'] = ann.pop('text')
            if 'start' in ann:
                ann['start_frame'] = ann.pop('start')
            if 'end' in ann:
                ann['end_frame'] = ann.pop('end')
        
        traj_name, _ = os.path.splitext(os.path.basename(af))
        annotations = sorted(annotations, key=lambda x: x['start_frame'] if x['start_frame'] is not None else float('inf'))

        if combine:
            # Combine labelled segments
            max_gap = 15
            first_success = 0
            while first_success < len(annotations) and not annotations[first_success]['success']:
                first_success += 1
            if first_success == len(annotations):
                print('No success found in', af)
                return

            seg_idxs = [[first_success, first_success]]
            for i, ann in enumerate(annotations[first_success+1:]):
                if not ann['success'] or ann['start_frame'] is None or ann['end_frame'] is None:
                    continue
                if annotations[seg_idxs[-1][1]]['end_frame'] + max_gap >= ann['start_frame']:
                    seg_idxs[-1][1] = first_success + i + 1
                else:
                    seg_idxs.append([i + first_success + 1, i + first_success + 1])

            for (start_idx, end_idx) in seg_idxs:
                start_frame = annotations[start_idx]['start_frame']
                end_frame = annotations[end_idx]['end_frame']
                if start_frame is None or end_frame is None:
                    continue
                labelled_traj_name = '{}_{:05d}-{:05d}'.format(traj_name, start_frame, end_frame)
                os.makedirs(os.path.join(labelled_data_dir, labelled_traj_name), exist_ok=True)
                current_seg = start_idx
                for i, s in enumerate(range(start_frame, end_frame)):
                    data = joblib.load(os.path.join(traj_dir, traj_name, '{:04d}.pkl'.format(s)))
                    data['task'] = annotations[current_seg]['language']
                    joblib.dump(data, os.path.join(labelled_data_dir, labelled_traj_name, '{:04d}.pkl'.format(i)), compress=3)
                    if s == annotations[current_seg]['end_frame']:
                        current_seg += 1
                
                open(os.path.join(labelled_data_dir, labelled_traj_name, 'success.txt'), 'w').close()
                metadata = {
                    'success': annotations[end_idx]['success'],
                }
                with open(os.path.join(labelled_data_dir, labelled_traj_name, 'metadata.json'), 'w') as f:
                    json.dump(metadata, f, indent=4)

        else:
            # Save each labelled segment as a separate trajectory
            for ann in annotations:
                start, end = ann['start_frame'], ann['end_frame']
                if start is None or end is None:
                    continue
                labelled_traj_name = f'{traj_name}_{ann["start_frame"]}-{ann["end_frame"]}'
                os.makedirs(os.path.join(labelled_data_dir, labelled_traj_name), exist_ok=True)
                for s,d in zip(range(start, end), range(0, end - start)):
                    src = os.path.join(traj_dir, traj_name, '{:04d}.pkl'.format(s))
                    dest = os.path.join(labelled_data_dir, labelled_traj_name, '{:04d}.pkl'.format(d))
                    os.system(f"cp {src} {dest}")
                
                if ann['success']:
                    open(os.path.join(labelled_data_dir, labelled_traj_name, 'success.txt'), 'w').close()
                else:
                    open(os.path.join(labelled_data_dir, labelled_traj_name, 'failure.txt'), 'w').close()
                
                metadata = {
                    'success': ann['success'],
                    'task': ann['language'],
                }
                with open(os.path.join(labelled_data_dir, labelled_traj_name, 'metadata.json'), 'w') as f:
                    json.dump(metadata, f, indent=4)
    except Exception as e:
        print("Error at", af)
        print(e)
        traceback.print_exc()

@click.command()
@click.option("--annotation-dir", required=True, type=str, help="Directory containing annotation jsons for each trajectory")
@click.option("--traj-dir", required=True, type=str, help="Directory containing training data in pkl format")
@click.option("--combine", is_flag=True, help="Combine all labelled segments into a single trajectory. Instruction labels will be saved per pkl in final trajectory.")
def organize(annotation_dir, traj_dir, combine):
    data_root, data_name = os.path.split(os.path.normpath(traj_dir))
    
    prefix = 'labelled-'
    if combine:
        prefix += 'combined-'
    labelled_data_dir = os.path.join(data_root, prefix + data_name)
    print(f'creating', labelled_data_dir)
    os.makedirs(labelled_data_dir, exist_ok=True)

    annotation_files = glob(os.path.join(annotation_dir, '*.json'))
    args = [
        (af, combine, traj_dir, labelled_data_dir) for af in annotation_files
    ]
    print(args)
    
    num_workers = 8
    with Pool(num_workers) as p:
        for _ in tqdm(p.imap(organize_seg, args), total=len(args)):
            pass
        

if __name__ == '__main__':
    organize()

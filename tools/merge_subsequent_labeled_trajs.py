import os
import shutil
import numpy as np
import json
import h5py
import joblib


def load_metadata(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)


def save_metadata(metadata, filepath):
    with open(filepath, 'w') as f:
        json.dump(metadata, f)


def copy_pkl_files(src_dir, dst_dir, start_frame, instruction, instruction_embedding):
    for filename in os.listdir(src_dir):
        if filename.endswith('.pkl'):
            frame_index = int(filename.split('.')[0]) + start_frame
            new_filename = f'{frame_index:04d}.pkl'
            shutil.copyfile(os.path.join(src_dir, filename), os.path.join(dst_dir, new_filename))

            with open(os.path.join(dst_dir, new_filename), 'rb') as f:
                data = joblib.load(f)
            data['instruction'] = instruction
            data['instruction_embedding'] = instruction_embedding
            with open(os.path.join(dst_dir, new_filename), 'wb') as f:
                joblib.dump(data, f, compress=3)


def concatenate_features(h5_files):
    arrays = []
    for h5_file in h5_files:
        with h5py.File(h5_file, 'r') as f:
            arrays.append(f['features'][:])
    return np.concatenate(arrays, axis=0)


def process_trajectories(base_dir, save_dir):
    trajectories = os.listdir(base_dir)
    grouped_trajectories = {}

    for traj in trajectories:
        date, time, frame_range = traj.split('_')
        start_frame, end_frame = map(int, frame_range.split('-'))
        date_key = f'{date}_{time}'

        if date_key not in grouped_trajectories:
            grouped_trajectories[date_key] = []

        grouped_trajectories[date_key].append((start_frame, end_frame, traj))

    for date_key, traj_list in grouped_trajectories.items():
        traj_list.sort()
        grouped = []
        current_group = [traj_list[0]]

        for i in range(1, len(traj_list)):
            if traj_list[i][0] == traj_list[i - 1][1]:
                current_group.append(traj_list[i])
            else:
                break

        grouped.append(current_group)

        for group in grouped:

            success = True
            for _, __, traj in group:
                if os.path.exists(os.path.join(base_dir, traj, 'failure.txt')):
                    success = False
                    break
            if not success:
                continue

            new_dir = os.path.join(save_dir, date_key)
            os.makedirs(new_dir, exist_ok=True)

            metadata = load_metadata(os.path.join(base_dir, group[0][2], 'metadata.json'))
            instruction = metadata.pop('instruction')
            instruction_embedding = metadata.pop('instruction_embedding')
            metadata.pop('task', None)
            save_metadata(metadata, os.path.join(new_dir, 'metadata.json'))

            h5_files = []
            frame_offset = 0

            for start_frame, end_frame, traj in group:
                traj_dir = os.path.join(base_dir, traj)

                metadata = load_metadata(os.path.join(traj_dir, 'metadata.json'))
                instruction = metadata['instruction']
                instruction_embedding = metadata['instruction_embedding']

                copy_pkl_files(traj_dir, new_dir, frame_offset, instruction, instruction_embedding)
                frame_offset += (end_frame - start_frame)

                h5_files.append(os.path.join(traj_dir, 'features.h5'))

            concatenated_features = concatenate_features(h5_files)
            with h5py.File(os.path.join(new_dir, 'features.h5'), 'w') as f:
                f.create_dataset('features', data=concatenated_features)

            if success:
                shutil.copyfile(os.path.join(base_dir, group[0][2], 'success.txt'),
                                os.path.join(new_dir, 'success.txt'))
            else:
                open(os.path.join(new_dir, "failure.txt"), 'w').close()


if __name__ == "__main__":
    base_directory = '/home/bfshi/data/bimanual/features/features_all/vitl-mae-egosoup/labelled-sort-yellowleft-blueright_04-06-2024'
    new_directory = '/home/bfshi/data/bimanual/features/features_all/vitl-mae-egosoup/merged-labelled-sort-yellowleft-blueright_04-06-2024'
    process_trajectories(base_directory, new_directory)

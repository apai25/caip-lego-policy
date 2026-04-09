import os
import numpy as np
import joblib
import h5py
import shutil
import argparse
from multiprocessing import Pool, cpu_count

def process_subfolder(args):
    subdir, root_folder, new_root_folder = args
    pkl_files = [file for file in os.listdir(subdir) if file.endswith(".pkl")]
    pkl_files = sorted(pkl_files)

    # read all the features
    features = []
    for file in pkl_files:
        pkl_file_path = os.path.join(subdir, file)
        # Load the .pkl file
        data = joblib.load(pkl_file_path)
        # Extract the arrays
        feat_left = data['feat_left']
        feat_hand = data['feat_hand']
        feat_right = data['feat_right']
        # Merge the arrays
        merged_array = np.stack([feat_left, feat_hand, feat_right], axis=0)
        features.append(merged_array)
    features = np.stack(features, axis=0)

    # Determine the new subdirectory path in the new root folder
    new_subdir = subdir.replace(root_folder, new_root_folder, 1)
    os.makedirs(new_subdir, exist_ok=True)

    # Save the features to a single HDF5 file
    hdf5_file_path = os.path.join(new_subdir, 'features.h5')
    with h5py.File(hdf5_file_path, 'w') as hdf_file:
        hdf_file.create_dataset('features', data=features)

    # Copy non-pkl files
    for item in os.listdir(subdir):
        source_path = os.path.join(subdir, item)
        destination_path = source_path.replace(root_folder, new_root_folder, 1)
        if not os.path.exists(destination_path):
            if item.endswith('.pkl'):  # remove features from pkl files
                data = joblib.load(source_path)
                data.pop('feat_left')
                data.pop('feat_hand')
                data.pop('feat_right')
                joblib.dump(data, destination_path)
            else:  # copy other files
                shutil.copy2(source_path, destination_path)

def main(root_folder, new_root_folder):
    # List all subdirectories in the root folder
    subdirs = [os.path.join(root_folder, subdir) for subdir in next(os.walk(root_folder))[1]]
    # Prepare arguments for multiprocessing
    args = [(subdir, root_folder, new_root_folder) for subdir in subdirs]

    # Use multiprocessing to process each subdirectory in parallel
    with Pool(processes=cpu_count() // 8) as pool:
        pool.map(process_subfolder, args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge arrays from pkl files and save as HDF5.")
    parser.add_argument("root_folder", type=str, help="Path to the root folder containing subfolders with pkl files.")
    parser.add_argument("new_root_folder", type=str, help="Path to the new root folder where the results will be saved.")
    args = parser.parse_args()

    main(args.root_folder, args.new_root_folder)

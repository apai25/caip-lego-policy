import os
import json
import joblib
import argparse
from tqdm import tqdm


def edit_pkl_files(root_directory, instruction_key, get_text_embedding):
    # Initialize a list to hold all the paths that will be processed
    paths_to_process = []

    # First, walk through the directory to gather all the paths
    for dirpath, _, filenames in os.walk(root_directory):
        for filename in filenames:
            if filename.endswith(".pkl"):
                full_path = os.path.join(dirpath, filename)
                paths_to_process.append(full_path)

    # Now process each file, with tqdm showing the progress
    for full_path in tqdm(paths_to_process, desc="Processing files"):

        # Read the current file
        content = joblib.load(full_path)

        if instruction_key in content:
            # Extract text embedding for the 'instruction'
            embedding_vector = get_text_embedding(content[instruction_key])

            # Update the content with the new 'instruction_embedding' vector
            content["instruction_embedding"] = embedding_vector.tolist()

            # Save the updated content back to the file
            joblib.dump(content, full_path, compress=3)


def edit_json_files(root_directory, instruction_key, get_text_embedding):
    # Initialize a list to hold all the paths that will be processed
    paths_to_process = []

    # First, walk through the directory to gather all the paths
    for dirpath, _, filenames in os.walk(root_directory):
        for filename in filenames:
            if filename == "metadata.json":
                full_path = os.path.join(dirpath, filename)
                paths_to_process.append(full_path)

    # Now process each file, with tqdm showing the progress
    for full_path in tqdm(paths_to_process, desc="Processing files"):

        # Read the current file
        with open(full_path, 'r') as file:
            content = json.load(file)

        if instruction_key in content:
            # Extract text embedding for the 'instruction'
            embedding_vector = get_text_embedding(content[instruction_key])

            # Update the content with the new 'instruction_embedding' vector
            content["instruction_embedding"] = embedding_vector.tolist()

            # Save the updated content back to the file
            with open(full_path, 'w') as file:
                json.dump(content, file, indent=4)


def main():
    parser = argparse.ArgumentParser(description='Process metadata.json files to add text embeddings.')
    parser.add_argument('--path', type=str,
                        help='Path to the root directory where the search should start.',
                        default='/home/niudt/data/bimanual_1001/features/features_attn_pooled_visual_feature_selector/dinov2-b/pick-yellow-right_04-29-2024')
    parser.add_argument('--instruction_key', type=str, default='instruction')
    parser.add_argument('--file_type', type=str, default='json')
    parser.add_argument('--language_model', type=str, default='t5', choices=['imagebind', 't5', 'clip'])

    args = parser.parse_args()

    if args.language_model == 'imagebind':
        from mvp.language_model.imagebind_language_model import get_text_embedding
    elif args.language_model == 't5':
        from mvp.language_model.sentence_t5 import get_text_embedding
    elif args.language_model == 'clip':
        from mvp.language_model.clip_language_model import get_text_embedding

    if args.file_type == 'json':
        edit_json_files(args.path, args.instruction_key, get_text_embedding)
    elif args.file_type == 'pkl':
        edit_pkl_files(args.path, args.instruction_key, get_text_embedding)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()
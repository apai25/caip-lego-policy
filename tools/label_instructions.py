import os
import json
import argparse


def label_instruction_stack(root_directory):
    for dirpath, _, filenames in os.walk(root_directory):
        for filename in filenames:
            if filename == "metadata.json":
                full_path = os.path.join(dirpath, filename)

                # Read the current file
                with open(full_path, 'r') as file:
                    content = json.load(file)
                    object_value = content.get("object")
                    target_value = content.get("target")

                    # Add or Update the "instruction" key
                    new_instruction = f"Put {object_value} on top of {target_value}."
                    content["instruction"] = new_instruction

                # Save the file with new content
                with open(full_path, 'w') as file:
                    json.dump(content, file, indent=4)


def label_instruction_pick(root_directory):
    for dirpath, _, filenames in os.walk(root_directory):
        for filename in filenames:
            if filename == "metadata.json":
                full_path = os.path.join(dirpath, filename)

                # Read the current file
                with open(full_path, 'r') as file:
                    content = json.load(file)
                    object_value = content.get("object")

                    # Add or Update the "instruction" key
                    new_instruction = f"Pick up {object_value}."
                    content["instruction"] = new_instruction

                # Save the file with new content
                with open(full_path, 'w') as file:
                    json.dump(content, file, indent=4)

def label_instruction(root_directory, instruction):
    for dirpath, _, filenames in os.walk(root_directory):
        for filename in filenames:
            if filename == "metadata.json":
                full_path = os.path.join(dirpath, filename)

                # Read the current file
                with open(full_path, 'r') as file:
                    content = json.load(file)
                    content["instruction"] = instruction

                # Save the file with new content
                with open(full_path, 'w') as file:
                    json.dump(content, file, indent=4)


def main():
    parser = argparse.ArgumentParser(description='Edit metadata.json files to add instructions.')
    parser.add_argument('--path', type=str,
                        help='Path to the root directory where the search should start.',
                        default='/home/niudt/data/bimanual_1001/features/features_attn_pooled_visual_feature_selector/dinov2-b/pick-yellow-right_04-29-2024'
                        )
    parser.add_argument('--task', type=str, default='any', choices=['any', 'stack', 'pick'], help='What is the task in the demos')
    parser.add_argument('--instruction', type=str, help='Instruction to label', default='pick right arm')

    args = parser.parse_args()

    if args.task == 'any':
        label_instruction(args.path, args.instruction)
    elif args.task == 'stack':
        label_instruction_stack(args.path)
    elif args.task == 'pick':
        label_instruction_pick(args.path)
    else:
        raise NotImplementedError



if __name__ == "__main__":
    main()

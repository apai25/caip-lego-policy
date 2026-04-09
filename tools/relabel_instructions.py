import os
import json
import argparse


def relabel_instruction(root_directory, instruction_mapping):
    traj_wo_instruction = []
    for dirpath, _, filenames in os.walk(root_directory):
        for filename in filenames:
            if filename == "metadata.json":
                full_path = os.path.join(dirpath, filename)

                # Read the current file
                with open(full_path, 'r') as file:
                    content = json.load(file)
                    if content["task"] in instruction_mapping.keys():
                        content["instruction"] = instruction_mapping[content["task"]]
                    elif content["task"] in instruction_mapping.values():
                        content["instruction"] = content["task"]
                    else:
                        traj_wo_instruction.append(dirpath)

                # Save the file with new content
                with open(full_path, 'w') as file:
                    json.dump(content, file, indent=4)

    return traj_wo_instruction


def label_traj_wo_instruction_as_fail(traj_wo_instruction):
    for directory in traj_wo_instruction:
        # Check if success.txt exists in the directory
        success_file = os.path.join(directory, 'success.txt')
        if os.path.isfile(success_file):
            # Delete success.txt
            os.remove(success_file)
            # Create failure.txt
            failure_file = os.path.join(directory, 'failure.txt')
            open(failure_file, 'a').close()  # Creating an empty failure file



def main():
    parser = argparse.ArgumentParser(description='Edit metadata.json files to add instructions.')
    parser.add_argument('--path', type=str,
                        help='Path to the root directory where the search should start.')

    args = parser.parse_args()

    # instruction_mapping = {
    #     "1": "Pick up the blue cube with left hand.",
    #     "2": "Place the blue cube in left hand on the blue tray.",
    #     "3": "Pick up the yellow cube with right hand.",
    #     "4": "Place the yellow cube in right hand on the yellow tray."
    # }

    instruction_mapping = {
        "1": "Pick up the yellow cube with left hand.",
        "2": "Place the yellow cube in left hand on the yellow tray.",
        "3": "Pick up the blue cube with right hand.",
        "4": "Place the blue cube in right hand on the blue tray."
    }

    traj_wo_instruction = relabel_instruction(args.path, instruction_mapping)
    label_traj_wo_instruction_as_fail(traj_wo_instruction)



if __name__ == "__main__":
    main()

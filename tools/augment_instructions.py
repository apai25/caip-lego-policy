import os
import json
from openai import OpenAI
import argparse
from tqdm import tqdm

client = OpenAI()

def paraphrase_instruction(instruction, temperature=1.0):
    """
    Function to paraphrase an instruction using GPT-3.5
    """
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  # You can choose the model version appropriate for your use case
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Rewrite this instruction with a different wording for variety: '{instruction}'"}
        ],
        temperature=temperature,
        max_tokens=60
    )
    return response.choices[0].message.content.strip()


def process_folder(root_folder):
    """
    Process each metadata.json in subfolders of the root folder
    """
    # First, gather all metadata.json file paths
    json_files = []
    for subdir, dirs, files in os.walk(root_folder):
        for filename in files:
            if filename == 'metadata.json':
                json_files.append(os.path.join(subdir, filename))

    # Process each file with a progress bar
    for file_path in tqdm(json_files, desc="Processing files"):
        with open(file_path, 'r+') as file:
            data = json.load(file)
            original_instruction = data.get('instruction', '')
            # Paraphrase the instruction
            paraphrased_instruction = paraphrase_instruction(original_instruction)
            data['instruction'] = paraphrased_instruction
            # Write the new instruction back to the file
            file.seek(0)  # Reset file pointer to the beginning
            json.dump(data, file, indent=4)
            file.truncate()  # Remove remaining part of the original file content

def main():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(
        description="Paraphrase instructions in metadata.json files within a directory structure.")
    parser.add_argument("--path", type=str,
                        help="The path to the root folder containing subfolders with metadata.json files.")
    args = parser.parse_args()

    process_folder(args.path)


if __name__ == "__main__":
    main()
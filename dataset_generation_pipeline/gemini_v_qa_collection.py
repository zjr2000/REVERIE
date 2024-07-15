import json
import random
import os
from multiprocessing.pool import Pool
import argparse
import google.generativeai as genai
from PIL import Image
import ast


PROMPT = """As an image-based question-answer generator, your task is to create 3 challenging questions related to a given image, each accompanied by one correct and one misleading answer. Ensure adherence to these guidelines:
1. Formulate questions that can be definitively answered using the image and general knowledge
2. The questions should require multi-step analysis to solve, avoiding overly simple queries
3. Draw inspiration from diverse fields such as daily life, mathematics, sciences or programming, focusing on visual elements within the image like objects, symbols, or text.
4. Employ a diverse array of question formats, such as open-ended, multiple-choice, yes/no, and short-answer types.
5. Provide one accurate answer and one plausible but incorrect answer for each question
6. Keep questions clear and concise
7. Present your output in JSON format, structured as follows: [{question1, correct answer1, confusing answer1}, {question2, correct answer2, confusing answer2}, ...]. Refrain from including extraneous content
"""

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_folder", type=str, default='dataset_augmentation/gemini_reasoning_instructions', help="Save folder")
    parser.add_argument("--image_folder", type=str, required=True, help="image folder")
    parser.add_argument("--api_key", required=True)
    parser.add_argument("--num_tasks", type=int, default=1, help="Number of splits.")
    args = parser.parse_args()
    return args


def generate(ori_file_paths, file_names, output_dir):
    for file in file_names:
        key = int(file[:-5])
        try:
            image = Image.open(ori_file_paths[key])
            model = genai.GenerativeModel('gemini-pro-vision')
            response = model.generate_content([image, PROMPT])
            raw_content = response.text
            formatted_content = ast.literal_eval(raw_content)
            formatted_content = {'image': ori_file_paths[key], 'content': formatted_content}
            print(ori_file_paths[key], key)
            with open(os.path.join(output_dir, file), 'w') as f:
                json.dump(formatted_content, f)
        except Exception as e:
            print(f"Error processing file '{ori_file_paths[key]}': {e}")



if __name__ == '__main__':
    args = parse_args()
    genai.configure(api_key=args.api_key)
    target_folder = args.target_folder
    output_dir = os.path.join(target_folder, 'reasoning_instruct_raw_data')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    with open(os.path.join(target_folder, 'image_list.json'), 'r') as f:
        images = json.load(f)
    images = [os.path.join(args.image_folder, i) for i in images]
    generated_result_files = [f"{_id}.json" for _id, _ in enumerate(images)]
    num_tasks = args.num_tasks  
    while True:
        try:

            # Files that have not been processed yet.
            completed_files = os.listdir(output_dir)
            print(f"completed_files: {len(completed_files)}")
            completed_files = set(completed_files)
            # Files that have not been processed yet.
            incomplete_files = [f for f in generated_result_files if f not in completed_files]
            print(f"incomplete_files: {len(incomplete_files)}")

            # Break the loop when there are no incomplete files
            if len(incomplete_files) == 0:
                break
            if len(incomplete_files) <= num_tasks:
                num_tasks = 1

            # Split tasks into parts.
            part_len = len(incomplete_files) // num_tasks
            all_parts = [incomplete_files[i:i + part_len] for i in range(0, len(incomplete_files), part_len)]
            task_args = [(images, part, output_dir) for part in all_parts]

            # Use a pool of workers to process the files in parallel.
            with Pool() as pool:
                pool.starmap(generate, task_args)
            break
        except Exception as e:
            print(f"Error: {e}")
    combined_contents = []

    # Iterate through json files
    for file_name in os.listdir(output_dir):
        if file_name.endswith(".json"):
            file_path = os.path.join(output_dir, file_name)
            with open(file_path, "r") as json_file:
                content = json.load(json_file)
            image = content['image']
            for item in content['content']:
                try:
                    item = {
                        'image': image,
                        'question': item['question'],
                        'correct_answer': item['correct_answer'],
                        'confusing_answer': item['confusing_answer']
                    }
                    combined_contents.append(item)
                except Exception as e:
                    print(f"Error: {e}")
                    continue
    with open(os.path.join(target_folder, 'reasoning_instruct_data.json'), 'w', encoding='utf-8') as f:
        json.dump(combined_contents, f, ensure_ascii=False)
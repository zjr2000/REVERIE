import json
import random
import os
from multiprocessing.pool import Pool
import argparse
import google.generativeai as genai
from PIL import Image
import ast


PROMPT_CORRECT = """You are an image-based question-answer analyst. Given an image and an image-based question-answer pair, your task is to generate rationale that leading to the correct answer. 
Ensure adherence to these guidelines:

1. Analyze the provided image carefully, focusing on key details relevant to the question.
2. The rationale should be concise and logic coherent. 
3. Ensure that each reasoning in the rationale is clear, and directly contributes to reaching the answer. Where appropriate, bring in external knowledge to provide context or clarify connections.
4. Do not repeat the correct answer. Output the rationale only. Refrain from including extraneous content

Question: {}
Answer: {}
Rationale:"""

PROMPT_INCORRECT = """You are tasked with analyzing an image in relation to a specific question and an incorrect answer provided for that question. Your main objective is to identify and explain why the answer is incorrect by closely examining the image and focusing on the crucial elements related to the question. Your analysis should

1. Analyze the provided image carefully, focusing on key details relevant to the question.
2. Craft a comprehensive rationale that explicates the reasons why the provided answer to the question is incorrect. This should involve a meticulous breakdown of the image's details, highlighting specific aspects that demonstrate the inaccuracy of the answer.
3. Ensure that each point in your rationale is clear, concise, and directly contributes to explaining why the answer is incorrect. The rationale should be logically structured, with each argument clearly supporting the conclusion that the answer is incorrect. Where appropriate, bring in external knowledge to provide context or clarify connections. 
4. Output the rationale only. Refrain from including extraneous content

Question: {}
Incorrect Answer: {}
Rationale:"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_folder", type=str, default='dataset_augmentation/gemini_reasoning_instructions')
    parser.add_argument("--api_key", required=True)
    parser.add_argument("--num_tasks", type=int, default=1, help="Number of splits.")
    args = parser.parse_args()
    return args


def generate(qa_data, file_names, output_dir):
    for file in file_names:
        key = int(file[:-5])
        sample = qa_data[key]
        question = sample['question']
        correct_answer = sample['correct_answer']
        confusing_answer = sample['confusing_answer']
        image_path = sample['image']
        image = Image.open(image_path)
        try:
            model = genai.GenerativeModel('gemini-pro-vision')
            correct_rationale = model.generate_content([image, PROMPT_CORRECT.format(question, correct_answer)])
            correct_rationale = correct_rationale.text
            incorrect_rationale = model.generate_content([image, PROMPT_INCORRECT.format(question, confusing_answer)])
            incorrect_rationale = incorrect_rationale.text
            formatted_content = {
                'image': image_path,
                'question': question,
                'correct_answer': correct_answer,
                'confusing_answer': confusing_answer,
                'correct_rationale': correct_rationale, 
                'incorrect_rationale': incorrect_rationale
            }
            print(image_path, question, key)
            with open(os.path.join(output_dir, file), 'w') as f:
                json.dump(formatted_content, f)
        except Exception as e:
            print(f"Error processing file '{image_path}': {e}")



if __name__ == '__main__':
    args = parse_args()
    genai.configure(api_key=args.api_key)
    target_folder = args.target_folder
    output_dir = os.path.join(target_folder, 'rationale_raw_data')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    with open(os.path.join(target_folder, 'reasoning_instruct_data.json'), 'r') as f:
        qa_data = json.load(f)
    
    generated_result_files = [f"{_id}.json" for _id, _ in enumerate(qa_data)]
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
            task_args = [(qa_data, part, output_dir) for part in all_parts]

            # Use a pool of workers to process the files in parallel.
            with Pool() as pool:
                pool.starmap(generate, task_args)
        except Exception as e:
            print(f"Error: {e}")
        break
    combined_contents = []

    # Iterate through json files
    for file_name in os.listdir(output_dir):
        if file_name.endswith(".json"):
            file_path = os.path.join(output_dir, file_name)
            with open(file_path, "r") as json_file:
                content = json.load(json_file)
            image = content['image']
            item = {
                'image': image,
                'question': content['question'],
                'correct_answer': content['correct_answer'],
                'confusing_answer': content['confusing_answer'],
                'correct_rationale': content['correct_rationale'],
                'incorrect_rationale': content['incorrect_rationale']
            }
            combined_contents.append(item)
    with open(os.path.join(target_folder, 'rationale_instruct_data.json'), 'w', encoding='utf-8') as f:
        json.dump(combined_contents, f, ensure_ascii=False)
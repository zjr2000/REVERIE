import json
import random
import os
from multiprocessing.pool import Pool
import argparse
import openai
import google.generativeai as genai



PROMPT = """Your task is to review a set of materials related to a specific question. These materials include a question, a positive answer with its analysis, and a negative answer with its analysis. Your primary goal is to determine whether there is a contradiction between the positive answer analysis and the negative answer analysis.

Materials Provided:

Question: The main question being addressed.
Positive Answer: The answer that correctly addresses the question.
Positive Answer Analysis: Explanation of why the positive answer is correct.
Negative Answer: A confusing and incorrect answer to the question.
Negative Answer Analysis: Analysis of why the negative answer is incorrect.

Steps to Follow:

1. Understand the question being asked.
2. Read the answer considered correct. Understand the explanation provided for why this answer is correct.
3. Read the answer considered incorrect. Understand the explanation provided for why this answer is incorrect.
4. Compare the reasons provided in both analyses. Identify specific instances where the analyses directly contradict each other beyond their natural stance of opposition. A contradiction exists if the rationale in the negative answer analysis conflicts with the postive answer analysis/postive answer. Contradictions may include conflicting facts or incompatible conclusions drawn from the same premise.
5. Note that the positive analysis aims to justify the correctness of the positive answer, while the negative analysis focuses on highlighting the flaws in the negative answer. This natural dichotomy is not the contradiction of interest.
6. Decide if a contradiction is present: Respond 'No' if there is no contradiction. Respond 'Yes' if there is a contradiction between the analyses.
7. Only output 'Yes' or 'No'. Refrain from including extraneous content

Question: {}
Positive Answer: {}
Positive Answer Analysis: {}
Negative Answer: {}
Negative Answer Analysis: {}"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_folder", type=str, default='dataset_augmentation/gemini_reasoning_instructions')
    parser.add_argument("--gemini_api_key", required=True)
    parser.add_argument("--chatgpt_api_key", required=True)
    parser.add_argument("--num_tasks", type=int, default=1, help="Number of splits.")
    args = parser.parse_args()
    return args


def request_gemini(prompt):
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(prompt)
    print('Gemini: {}'.format(response.text))
    return response.text


def request_chatgpt(prompt):
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-1106",
        messages=[        
            {
                "role": "user",
                "content": prompt                    
            }]
    )
    response = completion["choices"][0]["message"]["content"]
    print('ChatGPT: {}'.format(response))
    return response

def process(response):
    temp = response.lower()
    if temp.startswith('yes'):
        return 'yes'
    elif temp.startswith('no'):
        return 'no'
    else:
        raise ValueError(f"Invalid response: {response}")

def generate(qa_data, file_names, output_dir):
    for file in file_names:
        key = int(file[:-5])
        sample = qa_data[key]
        image = sample['image']
        question = sample['question']
        correct_answer = sample['correct_answer']
        confusing_answer = sample['confusing_answer']
        correct_rationale = sample['correct_rationale']
        incorrect_rationale = sample['incorrect_rationale']
        
        try:
            prompt = PROMPT.format(question, correct_answer, correct_rationale, confusing_answer, incorrect_rationale)
            gemini_response = process(request_gemini(prompt))
            # chatgpt_response = process(request_chatgpt(prompt))
            result = {
                'image': image,
                'question': question,
                'correct_answer': correct_answer,
                'confusing_answer': confusing_answer,
                'correct_rationale': correct_rationale,
                'incorrect_rationale': incorrect_rationale,
                'gemini_pro_judge': gemini_response,
                # 'gpt3.5_judge': chatgpt_response
            }
            print(image, gemini_response)
            
            with open(os.path.join(output_dir, file), 'w') as f:
                json.dump(result, f, indent=4)
        except Exception as e:
            print(f"Error processing file '{key}': {e}")


if __name__ == '__main__':
    args = parse_args()
    
    genai.configure(api_key=args.gemini_api_key)

    target_folder = args.target_folder
    output_dir = os.path.join(target_folder, 'check_results')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    with open(os.path.join(target_folder, 'rationale_instruct_data.json'), 'r') as f:
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
            try:
                image = content['image']
                if 'ocr_vqa' in image:
                    continue
                item = {
                    'image': image,
                    'question': content['question'],
                    'correct_answer': content['correct_answer'],
                    'confusing_answer': content['confusing_answer'],
                    'correct_rationale': content['correct_rationale'],
                    'incorrect_rationale': content['incorrect_rationale'],
                    'gemini_pro_judge': content['gemini_pro_judge'],
                    'chatgpt_judge': content['gpt3.5_judge']
                }
                combined_contents.append(item)
            except Exception as e:
                print(f"Error processing file '{file_name}': {e}")
                continue
    with open(os.path.join(target_folder, 'rationale_instruct_data_with_judge.json'), 'w', encoding='utf-8') as f:
        json.dump(combined_contents, f, ensure_ascii=False, indent=4)
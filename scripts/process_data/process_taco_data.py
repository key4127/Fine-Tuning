import os
import sys
import json
from datasets import load_dataset

sys.set_int_max_str_digits(100000)

def generate_taco_to_jsonl(sample):
    solutions = json.loads(sample["solutions"])
    if not solutions:
        return None
    
    user_prompt = sample['question']
    if sample['starter_code']:
        user_prompt = user_prompt + "You should continue to supplement based on the following code: " + sample['starter_code']
    if sample['input_output']:
        input_output = json.loads(sample['input_output'])
        if input_output['inputs'] and input_output['outputs']:
            user_prompt += f"\nExample inputs: {len(input_output['inputs'])} test cases: "
            user_prompt += "; ".join([f'input: "{inp}"' for inp in input_output['inputs']])
            user_prompt += f"\nExpected outputs: "
            user_prompt += "; ".join([f'output: "{out}"' for out in input_output['outputs']])

    conversation = [
        {"from": "system", "value": "You are a Python programming assistant. " \
                                    "You need to provide code that meets the user's requirements." \
                                    "Please carefully solve the following programming problem. When writing code, pay special attention to:" \
                                    "Edge Cases: Consider empty inputs, extreme values, and special situations\n" \
                                    "Problem Requirements: Read the output format and requirements carefully to ensure correct data types and formats are returned\n" \
                                    "Algorithm Selection: Choose the correct algorithm to avoid logical errors\n" \
                                    "Completeness: Ensure all possible input scenarios are handled\n" \
                                    "Code Robustness: Perform necessary input validation and error handling"
        },
        {"from": "user", "value": sample['question']},
        {"from": "assistant", "value": solutions[0]}
    ]

    return conversation

def process_taco_data(output_path):
    hf_token = os.environ.get("HUGGING_FACE_READ_DATASET_TOKEN")
    taco = load_dataset('BAAI/TACO', split='train', token=hf_token, trust_remote_code=True)

    with open(output_path, "w", encoding="utf-8") as output:
        for sample in taco:
            conversation = generate_taco_to_jsonl(sample)
            if conversation is not None:
                output.write(json.dumps(conversation, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    output_path = "./data/train/taco_data.jsonl"
    process_taco_data(output_path)
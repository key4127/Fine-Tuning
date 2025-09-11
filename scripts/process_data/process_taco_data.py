import os
import json
from datasets import load_dataset, load_from_disk

def generate_taco_to_jsonl(sample):
    solutions = json.loads(sample["solutions"])
    if not solutions:
        return None

    conversation = [
        {"from": "system", "value": "You are a Python programming assistant. " \
                                    "You need to provide code that meets the user's requirements."},
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
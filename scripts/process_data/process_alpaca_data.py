import json
import os

def process_alpaca_data(item):
    instruction = item.get('instruction', '')
    input_data = item.get('input', '')
    output_data = item.get('output', '')

    if input_data and (input_data != "< noinput >"):
        user_prompt = f"{instruction}\n\n```python\n{input_data}\n```"
    else:
        user_prompt = instruction

    assistant_response = f"```python\n{output_data}\n```"

    conversation = [
        {"from": "system", "value": "You are a Python programming assistant. " \
                                    "You need to provide code that meets the user's requirements"},
        {"from": "user", "value": user_prompt},
        {"from": "assistant", "value": assistant_response}
    ]

    return conversation

def convert_to_jsonl(input_file, output_file):
    """
    Processes the Alpaca data from the input file and writes it to the output file.
    
    Args:
        input_file (str): Path to the input JSON file containing Alpaca data.
        output_file (str): Path to the output JSONL file to write processed data.
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"The input file {input_file} does not exist.")
    
    with open(input_file, 'r', encoding='utf-8') as infile:
        data = json.load(infile)
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for item in data:
            conversation = process_alpaca_data(item)
            outfile.write(json.dumps(conversation, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    input_path = './data/origin_data/code_alpaca_20k.json'
    output_path = './data/train/alpaca_data.jsonl'
    
    try:
        convert_to_jsonl(input_path, output_path)
        print(f"Data processed successfully and saved to {output_path}")
    except Exception as e:
        print(f"An error occurred: {e}")
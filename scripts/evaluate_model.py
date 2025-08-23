import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from evaluate import load

def load_data(data_path: str):
    dataset = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            dataset.append(json.loads(line))
    return dataset

def extract_code(output: str) -> str:
    start_token = "```python"
    end_token = "```"

    if start_token in output:
        start_index = output.find(start_token) + len(start_token)
        end_index = output.find(end_token, start_index)
        if end_index != -1:
            return output[start_index:end_index].strip()
    return ""

def generate_code(model, tokenizer, prompt, max_new_tokens=512):
    full_prompt = [
        {"role": "system", "content": "You are a Python programming assistant. "\
        "You must write code that meets the user's requirements."\
        "Strictly return only the required code block, without any examples, comments, main program, or calls."},
        {"role": "user", "content": f"{prompt}"}
    ]

    tokenized_prompt = tokenizer.apply_chat_template(
        full_prompt,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    raw_codes_output = model.generate(
        tokenized_prompt,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=False,
        num_return_sequences=1
    )

    decoded_codes_output = tokenizer.decode(raw_codes_output[0], skip_special_tokens=True)

    return extract_code(decoded_codes_output)
    
def evaluate_with_data(model, tokenizer, data):
    predictions = []
    prompts = data["prompts"]
    print("begin generate codes\n")

    for prompt in prompts:
        code = generate_code(model, tokenizer, prompt)
        predictions.append(code)
        print(code + "\n")

def evaluate_MBPP(data_path: str, model, tokenizer):
    raw_data = load_data(data_path)

    data = {
        "prompts": [item["text"] for item in raw_data],
        "metric_name": "mbpp"
    }

    evaluate_with_data(model, tokenizer, data)

def evaluate_HumanEval(data_path: str, model, tokenizer):
    raw_data = load_data(data_path)

    data = {
        "prompts": [item["prompt"] for item in raw_data],
        "metric_name": "humaneval"
    }

    evaluate_with_data(model, tokenizer, data)

def evaluate_model(model_path: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    MBPP_path = "./data/test/mbpp.jsonl"
    HumanEval_path = "./data/test/human-eval.jsonl"

    # evaluate_MBPP(MBPP_path, model, tokenizer)
    evaluate_HumanEval(HumanEval_path, model, tokenizer)

if __name__ == "__main__":
    model = "./model/Qwen2.5-Coder-1.5B-Instruct/"
    evaluate_model(model)
import json
import torch
import time
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

def generate_codes(model, tokenizer, prompts, batch_size=4):
    print(f"模型所在设备: {model.device}")
    print(f"CUDA是否可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"当前GPU内存使用: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    results = []
    
    # 分批处理提示
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        print(f"处理批次 {i//batch_size + 1}/{(len(prompts)-1)//batch_size + 1}")
        
        # 为每个提示构建对话
        full_prompts = []
        for prompt in batch_prompts:
            messages = [
                {"role": "system", "content": "You are a Python programming assistant."},
                {"role": "user", "content": prompt}
            ]
            prompt_text = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            full_prompts.append(prompt_text)
        
        inputs = tokenizer(
            full_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
            padding_side='left'
        ).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,
                temperature=1.0,
                num_return_sequences=1
            )
        
        for j in range(len(batch_prompts)):
            input_length = inputs.input_ids[j].size(0)
            generated_text = tokenizer.decode(outputs[j][input_length:], skip_special_tokens=True)
            results.append(extract_code(generated_text))
    
    return results
    
def evaluate_with_data(model, tokenizer, data, k):
    prompts = data["prompts"]
    print("begin generate codes\n")

    codes = generate_codes(model, tokenizer, prompts)
    print(codes)

    #code_eval = load("code_eval")
    #test_cases = []
    #candidates = []
    #pass_at_k, results = code_eval.compute(
    #    references=test_cases, 
    #    predictions=candidates,
    #    k=ks
    #)

    #print(f"Pass_at_k: {pass_at_k}")
    #print(results)
    #print("\n")

def evaluate_MBPP(data_path: str, model, tokenizer):
    raw_data = load_data(data_path)

    data = {
        "prompts": [item["text"] for item in raw_data],
        "metric_name": "mbpp"
    }

    k = 1
    evaluate_with_data(model, tokenizer, data, k)

def evaluate_HumanEval(data_path: str, model, tokenizer):
    raw_data = load_data(data_path)

    data = {
        "prompts": [item["prompt"] for item in raw_data],
        "metric_name": "humaneval"
    }

    k = 1
    evaluate_with_data(model, tokenizer, data, k)

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
    start_time = time.time()
    evaluate_HumanEval(HumanEval_path, model, tokenizer)
    end_time = time.time()
    print(f"Evaluation time: {end_time - start_time} seconds")

if __name__ == "__main__":
    model = "./model/Qwen2.5-Coder-1.5B-Instruct/"
    evaluate_model(model)

# single: (k = 1) 2129.947891s
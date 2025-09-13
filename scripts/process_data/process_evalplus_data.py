import json

def contract(origin_input: str, merged_input: str, output: str):
    with open(origin_input, "r", encoding="utf-8") as input:
        origin_results = json.load(input)
    with open(merged_input, "r", encoding="utf-8") as input:
        merged_results = json.load(input)

    origin_results = origin_results.get("eval")
    merged_results = merged_results.get("eval")

    output_content = []

    for idx in range(len(origin_results)):
        origin = origin_results[f'HumanEval/{idx}'][0]
        merged = merged_results[f'HumanEval/{idx}'][0]
        
        if origin["base_status"] != "fail" and merged["base_status"] == "fail":
            data = {
                "task_id": origin["task_id"],
                "pass_solution": origin["solution"],
                "fail_solution": merged["solution"]
            }
            output_content.append(data)

    with open(output, "w", encoding="utf-8") as out:
        json.dump(output_content, out, ensure_ascii=False)

def main():
    origin_input_path = "./evalplus_results/humaneval/model--Qwen2.5-Coder-1.5B-Instruct_vllm_temp_0.0.eval_results.json"
    merged_input_path = "./evalplus_results/humaneval/model_output--QLora--random_data_r_4_vllm_temp_0.0.eval_results.json"
    output_path = "./evalplus_results/processed/humaneval.jsonl"

    contract(origin_input_path, merged_input_path, output_path)

if __name__ == "__main__":
    main()
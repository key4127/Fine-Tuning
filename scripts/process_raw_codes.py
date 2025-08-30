import json

def load_data(data_path: str):
    dataset = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            dataset.append(json.loads(line))
    return dataset

def process_MBPP_codes(origin_path: str, input_path: str, output_path: str):
    dataset = load_data(origin_path)

    with open(input_path, "r") as infile, open(output_path, "w") as outfile:
        for line, origin_data in zip(infile, dataset):
            processed_line = line.strip()
            generated_code = json.loads(processed_line)

            processed_data = {
                "code": [generated_code["codes"]],
                "tests": origin_data['test_list']
            }
            outfile.write(json.dumps(processed_data, ensure_ascii=False) + "\n")

def process_HumanEval_codes(origin_path: str, input_path: str, output_path: str):
    dataset = load_data(origin_path)

    with open(input_path, "r") as infile, open(output_path, "w") as outfile:
        for line, origin_data in zip(infile, dataset):
            processed_line = line.strip()
            code = json.loads(processed_line)

            processed_data = {
                "code": [code["codes"]],
                "tests": origin_data["test"]
            }
            outfile.write(json.dumps(processed_data, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    MBPP_origin_path = "./data/test/mbpp.jsonl"
    HumanEval_origin_path = "./data/test/human-eval.jsonl"
    MBPP_input_path = "./output/raw_mbpp1.jsonl"
    HumanEval_input_path = "./output/raw_human-eval1.jsonl"
    MBPP_output_path = "./output/mbpp1.jsonl"
    HumanEval_output_path = "./output/human-eval1.jsonl"

    process_MBPP_codes(
        MBPP_origin_path, 
        MBPP_input_path, 
        MBPP_output_path
    )
    process_HumanEval_codes(
        HumanEval_origin_path, 
        HumanEval_input_path, 
        HumanEval_output_path
    )
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

base_model_path = "./model/Qwen2.5-Coder-1.5B-Instruct/"
peft_model_path = "./model_output/QLora/peft_random_data_epoch_1/"
merged_model_save_path = "./model_output/QLora/random_data_epoch_1/"

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    dtype=torch.bfloat16,
    device_map="auto"
)

model = PeftModel.from_pretrained(
    base_model,
    peft_model_path
)

merged_model = model.merge_and_unload()
merged_model.save_pretrained(merged_model_save_path)

tokenizer = AutoTokenizer.from_pretrained(peft_model_path)
tokenizer.save_pretrained(merged_model_save_path)
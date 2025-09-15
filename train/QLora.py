from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from datasets import Dataset
import torch
import json
import random

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

model_path = "./model/Qwen2.5-Coder-1.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    device_map="auto"
)
model.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

data_paths = [
    "./data/train/alpaca_data.jsonl",
    "./data/train/taco_data.jsonl"
]
formatted_data = []

for data_path in data_paths:
    with open(data_path, "r", encoding="utf-8") as data:
        all_lines = data.readlines()

    total_lines = len(all_lines)
    selected_indices = random.sample(range(total_lines), total_lines // 2)
    selected_indices.sort()

    for idx in selected_indices:
        line = all_lines[idx].strip()
        record = json.loads(line)
        new_conv = [{"role": item["from"], "content": item["value"]} for item in record]
        formatted = tokenizer.apply_chat_template(
            new_conv, 
            tokenize=False, 
            add_generation_prompt=False
        )
        formatted_data.append({"messages": formatted})

qwen_dataset = Dataset.from_list(formatted_data)

training_args = SFTConfig(
    output_dir="./model_output/QLora",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    optim="paged_adamw_8bit",
    learning_rate=2e-4,
    fp16=True,
    logging_steps=100,
    save_strategy="epoch",
    max_length=512,
    dataset_text_field="messages"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=qwen_dataset,
    peft_config=lora_config,
    processing_class=tokenizer,
    args=training_args
)

print("begin training...")
trainer.train()

print("save training...")
trainer.save_model("./model_output/QLora/peft_learning_rate/")
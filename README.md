原模型：Qwen2.5-Coder-1.5B-Instruct

训练用数据集：code alpaca
测试用数据集：human-eval、MBPP

训练前：  

|MBPP|MBPP+|human-eval|human-eval+|
|:-:|:-:|:-:|:-:|
|0.685|0.590|0.695|0.640|

训练参数如下：  

```python
r=6
lora_alpha=32
target_modules=["q_proj", "v_proj"]
lora_dropout=0.05
bias="none"
num_train_epochs=3
per_device_train_batch_size=4
gradient_accumulation_steps=4
learning_rate=2e-4
max_length=512
```

机器显存为8G，实际训练占用6250MB。
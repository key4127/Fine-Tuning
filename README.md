原模型：Qwen2.5-Coder-1.5B-Instruct

训练用数据集：code alpaca、TACO  
测试用数据集：human-eval、MBPP

|训练情况|MBPP|MBPP+|bigcodebench|显存占用|
|:-:|:-:|:-:|:-:|:-:|
|训练前|0.685|0.590|0.255|/|
|random_epoch_1|0.714|0.624|0.257|/|
|prompt_epoch_1|0.712|0.624|0.263|7300MB|

random_epoch_1：  
训练从code alpaca和TACO各中随机选取一半数据  
具体参数如下  

```python
r=8, 
lora_alpha=16,
target_modules=["q_proj", "v_proj"],
lora_dropout=0.05,
num_train_epochs=1,
per_device_train_batch_size=4,
gradient_accumulation_steps=4,
learning_rate=1e-4
```

prompt_epoch_1在random的基础上改进了prompt。

机器显存为8G。

训练后在humanEval上表现较差，应该与prompt格式有关。所有训练数据集都采用自然语言，但humanEval采用docstring格式。
# evalplus.evaluate --model "./model/Qwen2.5-Coder-1.5B-Instruct" \
evalplus.evaluate --model "./model_output/QLora/learning_rate/" \
                  --dataset mbpp \
                  --backend vllm \
                  --greedy
# evalplus.evaluate --model "./model/Qwen2.5-Coder-1.5B-Instruct" \
evalplus.evaluate --model "./model_output/QLora/only_taco/" \
                  --dataset mbpp \
                  --backend vllm \
                  --greedy
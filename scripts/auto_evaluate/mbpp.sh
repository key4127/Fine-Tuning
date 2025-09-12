# evalplus.evaluate --model "./model/Qwen2.5-Coder-1.5B-Instruct" \
evalplus.evaluate --model "./model_output/QLora/random_data_epoch_1/" \
                  --dataset mbpp \
                  --backend vllm \
                  --greedy
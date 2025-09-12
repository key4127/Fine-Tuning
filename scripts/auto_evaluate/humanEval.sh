# evalplus.evaluate --model "./model/Qwen2.5-Coder-1.5B-Instruct" \
evalplus.evaluate --model "./model_output/QLora/random_data_r_4/" \
                  --dataset humaneval \
                  --backend vllm \
                  --greedy
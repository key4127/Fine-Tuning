bigcodebench.evaluate \
  --model "./model_output/QLora/half_both/" \
  --execution gradio \
  --split instruct \
  --subset full \
  --backend vllm
# --model "./model/Qwen2.5-Coder-1.5B-Instruct" \
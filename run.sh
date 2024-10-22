export CUDA_HOME="/usr/local/cuda-12.2"
export LIBRARY_PATH="/usr/local/cuda-12.2/lib64:$LIBRARY_PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH"


ACCELERATE_LOG_LEVEL=info accelerate launch --config_file accelerate_configs/deepspeed_zero3_cpu.yaml --mixed_precision bf16 \
    --num_processes 8 \
    tpo_train.py configs/config_full.yaml \
    --model_name_or_path="" \
    --data_path="" \
    --per_device_train_batch_size=2 \
    --gradient_accumulation_steps=8 \
    --torch_dtype=bfloat16 \
    --bf16=True \
    --beta=0.4 \
    --num_train_epochs=4 \
    --save_strategy='steps' \
    --save_steps=200 \
    --save_total_limit=1 \
    --output_dir="" \
    --prompt="qwen2-boxed"

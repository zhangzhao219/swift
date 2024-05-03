NPROC_PER_NODE=2

CUDA_VISIBLE_DEVICES=2,3 \
torchrun \
    --nproc_per_node=$NPROC_PER_NODE \
    --master_port 29501 \
    full_train.py \
    --dtype bf16 \
    --epochs 1 \
    --eval_batch_size 1 \
    --eval_steps 10 \
    --gpu_memory_fraction 1.0 \
    --learning_rate 1e-5 \
    --lr_scheduler_type linear \
    --max_length 1024 \
    --model_dir /root/swift/pretrained/google/gemma-2b-it \
    --output_dir output \
    --seed 42 \
    --template_type gemma \
    --train_data /root/swift/data/test.jsonl \
    --train_batch_size 1 \
    --save_steps 20 \
    --seed 42 \
    --val_data /root/swift/data/test.jsonl \
    --warmup_ratio 0.1 \
    --weight_decay 0.1

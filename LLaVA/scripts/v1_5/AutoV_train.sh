#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

deepspeed --master_port 30000 llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --train_type reward \
    --tune_prompt_selector True \
    --model_name_or_path ./checkpoints/llava-v1.5-7b \
    --version v1 \
    --data_path ../data/llava_v1_5_mix100k_reward.json \
    --image_folder ../data/attnmap100K \
    --is_reward True \
    --vision_tower ./checkpoints/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/autov_llava-v1.5-7b-reward-100k \
    --num_train_epochs 40 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 180000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.2 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb

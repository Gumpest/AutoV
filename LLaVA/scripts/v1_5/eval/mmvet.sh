#!/bin/bash
MODEL="llava-v1.5-7b"

python -m llava.eval.model_vqa \
    --model-path ./checkpoints/$MODEL \
    --question-file ./playground/data/eval/mm-vet/llava-mm-vet.jsonl \
    --image-folder ./playground/data/api_eval/mmvet/APICLIP_mmvet_ViT-L-14-336_23/1_3_BICUBIC_0 \
    --answers-file ./playground/data/eval/mm-vet/answers/$MODEL.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p ./playground/data/eval/mm-vet/results

python scripts/convert_mmvet_for_eval.py \
    --src ./playground/data/eval/mm-vet/answers/$MODEL.jsonl \
    --dst ./playground/data/eval/mm-vet/results/$MODEL.json


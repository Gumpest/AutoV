#!/bin/bash
MODEL='llava-v1.5-7b'

python -m llava.eval.model_vqa_loader \
    --model-path ./checkpoints/$MODEL \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --image-folder ./playground/data/redcircle/pope \
    --answers-file ./playground/data/eval/pope/answers/$MODEL.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_pope.py \
    --annotation-dir ./playground/data/eval/pope/coco \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --result-file ./playground/data/eval/pope/answers/$MODEL.jsonl

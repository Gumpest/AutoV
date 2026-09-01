#!/bin/bash
MODEL="llava-v1.5-7b"

python -m llava.eval.model_vqa_gemini \
    --model-path ./checkpoints/$MODEL \
    --question-file ./playground/data/eval/vizwiz/llava_test.jsonl \
    --image-folder ./playground/data/eval/vizwiz/test \
    --answers-file ./playground/data/eval/vizwiz/answers/gemini.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python scripts/convert_vizwiz_for_submission.py \
    --annotation-file ./playground/data/eval/vizwiz/llava_test.jsonl \
    --result-file ./playground/data/eval/vizwiz/answers/gemini.jsonl \
    --result-upload-file ./playground/data/eval/vizwiz/answers_upload/gemini.json

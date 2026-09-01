import json
import os
import torch
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)
import requests
from PIL import Image
from io import BytesIO
import re
import tqdm
from pathlib import Path

def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out

data_root = Path(os.environ.get("AUTOV_DATA_ROOT", Path(__file__).resolve().parents[3] / "data"))
data = json.load(open(data_root / 'llava_v1_5_mix100k_filter.json'))
LAYERS = [15, 20, 22, 23]

collection = []

model_path = "liuhaotian/llava-v1.5-7b"
model_name = get_model_name_from_path(model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path, None, model_name
)

for identity, item in enumerate(data):
    new_item = {}
    image_path = item['image']
    print(identity, " ", image_path)
    qs = item['conversations'][0]['value']
    label = item['conversations'][1]['value']
    idx_list =[]
    attn_path = []
    loss_list = []

    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    # else:
    #     if model.config.mm_use_im_start_end:
    #         qs = image_token_se + "\n" + qs
    #     else:
    #         qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    conv_mode = "vicuna_v1"

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    # 构造完整的 Prompt（Question + Label）
    full_prompt = prompt + " " + label  # 让 question 在前，label 在后

    for idx, l in enumerate(LAYERS):
        idx_list.append(idx)
        attn_path.append(f'APICLIP_llava_ViT-L-14-336_{l}/1_3_BICUBIC_0/{image_path}')
        img = data_root / 'attnmap100K' / f'APICLIP_llava_ViT-L-14-336_{l}' / '1_3_BICUBIC_0' / image_path
        image = load_image(img)
        image_size = [image.size]
        image_tensor = process_images(
            [image],
            image_processor,
            model.config
        ).to(model.device, dtype=torch.float16)

        input_ids = (
            tokenizer_image_token(full_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .cuda()
        )
        question_ids = (
            tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .cuda()
        )
        question_length = question_ids.shape[1]  # 获取 question token 长度

        # 重新 Tokenize `label`
        label_tokenized = tokenizer(label, return_tensors="pt", truncation=True, max_length=512)

        # 创建 labels，先填充 -100
        labels = torch.full_like(input_ids, -100)

        # 只复制 label 的部分
        label_length = label_tokenized.input_ids.shape[1]
        if question_length + label_length > input_ids.shape[1]:  # 防止超出 max_length
            label_length = input_ids.shape[1] - question_length

        labels[0, question_length:question_length + label_length] = label_tokenized.input_ids[0, -label_length:]  
        labels[labels == tokenizer.pad_token_id] = -100  # 忽略 pad token

        with torch.no_grad():
            outputs = model(input_ids, labels=labels.cuda(), images=image_tensor, image_sizes=image_size)
            loss = outputs.loss  # 交叉熵 loss
            # logits = outputs.logits  # 预测的 token 概率分布
        loss_list.append(loss.item())

    new_item['image'] = item['image']
    new_item['conversations'] = item['conversations']
    new_item['id'] = item['id']
    new_item['attn_map_path'] = attn_path
    new_item['attn_map_idx'] = idx_list
    new_item['llava_loss'] = loss_list

    collection.append(new_item)

with open(data_root / 'llava_v1_5_mix100k_reward.json', 'w') as f:
    json.dump(collection, f, indent=4)

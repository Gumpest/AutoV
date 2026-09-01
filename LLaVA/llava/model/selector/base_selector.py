import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F

import csv

def save_sorted_indices(sorted_indices, filename='output.csv'):
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(sorted_indices.tolist())


class FrameSelector(nn.Module):
    def __init__(self):
        super().__init__()

        self.fs_num_hidden_layers = 1

        # TODO: hard code
        self.resue_llm_layers = 1
        print(f'Reuse bottom {self.resue_llm_layers} layers of llm ...')

        # TODO: hard code
        self.hidden_size = 4096

        self.text_layers = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU()
        )
        self.image_layers = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU()
        )

        self.steps = 0
    

    def save_layers_weights(self, save_path):
        state_dict = {
            'text_layers': self.text_layers.state_dict(),
            'image_layers': self.image_layers.state_dict()
        }
        torch.save(state_dict, save_path)


    def load_layers_weights(self, load_path, device=None):
        state_dict = torch.load(load_path, map_location=device)
        self.text_layers.load_state_dict(state_dict['text_layers'])
        self.image_layers.load_state_dict(state_dict['image_layers'])


    def reward_forward(self, img_feature, text_feature, attn_map_idx, llava_loss, llm):
        """
        input
        img_feature: b*num_attn x num_img_tokens (196) x d (4096)
        text_feature: b*[num_text_tokens x d]
        """
        self.steps += 1
        # print('steps: ', self.steps)
        if self.steps % 1000 == 0:
            self.save_layers_weights(f'./checkpoints/autov_llava-v1.5-7b-reward-100k/selector_{self.steps}steps.pth')

        num_attn_img = len(attn_map_idx[0])
        norm = nn.LayerNorm(num_attn_img, elementwise_affine=False)
        device = img_feature.device
        all_ranking_loss = []

        for batch_idx, cur_text_feature in enumerate(text_feature):
            # cur_img_feature: num_attn x num_img_tokens x d, cur_text_feature: num_attn x num_text_tokens x d
            cur_img_feature = img_feature[batch_idx*num_attn_img: (batch_idx+1)*num_attn_img]
            cur_text_feature = cur_text_feature.unsqueeze(0).repeat(num_attn_img, 1, 1)
            cur_attn_map_idx = attn_map_idx[batch_idx]
            cur_llava_loss = llava_loss[batch_idx]

            hidden_states = torch.cat([cur_img_feature, cur_text_feature], dim=1) # [4, 642, 4096]

            attention_mask = None
            past_key_values_length = 0
            seq_length = hidden_states.shape[1]
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )

            position_ids = position_ids.unsqueeze(0).repeat(num_attn_img, 1)
            past_key_value = None

            # reuse llm
            with torch.no_grad():
                for _, decoder_layer in enumerate(llm[: self.resue_llm_layers]):
                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_value,
                        output_attentions=False,
                        use_cache=False,
                    )
                hidden_states = layer_outputs[0] # [4, 642, 4096]

            # 1. split text and image
            new_text_feature = hidden_states[:, cur_img_feature.shape[1]:, :]
            new_img_feature = hidden_states[:, :cur_img_feature.shape[1], :]

            new_text_feature = self.text_layers(new_text_feature) # [4, 66, 4096]
            new_img_feature = self.image_layers(new_img_feature) # [4, 576, 4096]

            # 2. compute simarity
            img_text_sim = torch.matmul(new_img_feature, new_text_feature.transpose(-2, -1)) # [4, 576, 66]

            # 3. compute weight between text and each attn_image
            combination_sim = []
            for i in range(len(cur_attn_map_idx)):
                combination_sim.append(torch.mean(img_text_sim[i]))
            combination_sim = torch.stack(combination_sim) # e.g., [0.8867, 0.9102, 0.8789, 0.9180]
            sorted_indices = [i for i, _ in sorted(enumerate(cur_llava_loss),
                                key=lambda x: x[1])] # e.g., [2, 1, 0, 3]
            combination_sim = combination_sim[sorted_indices] # sorted by llava_loss

            # 4. generate all combinations of 2 elements
            elements = range(1, len(cur_attn_map_idx) + 1)
            combinations = list(itertools.combinations(elements, 2))
            combinations = torch.tensor(combinations, dtype=torch.long, device=combination_sim.device) # [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]

            # 5. compute ranking loss
            cur_sample_ranking_loss = []
            for pairwise_combination in combinations:
                cur_pair = combination_sim[pairwise_combination - 1] # e.g., [0.8867, 0.9102]
                cur_pair_ranking_loss = -torch.log(torch.nn.functional.sigmoid(cur_pair[0] - cur_pair[1]))
                cur_sample_ranking_loss.append(cur_pair_ranking_loss)

            cur_sample_ranking_loss = torch.stack(cur_sample_ranking_loss).mean()
            all_ranking_loss.append(cur_sample_ranking_loss)
        all_ranking_loss = torch.stack(all_ranking_loss).mean()
        
        return all_ranking_loss


    @torch.no_grad()
    def reward_generate(self, img_feature, text_feature, llm):
        """
        input
        img_feature: b*num_attn x num_img_tokens (196) x d (4096)
        text_feature: b*[num_text_tokens x d]
        """
        num_attn_img = img_feature.shape[0]
        device = img_feature.device
        selected_img_feature = []

        for batch_idx, cur_text_feature in enumerate(text_feature):
            # cur_img_feature: num_attn x num_img_tokens x d, cur_text_feature: num_attn x num_text_tokens x d
            cur_img_feature = img_feature[batch_idx*num_attn_img: (batch_idx+1)*num_attn_img]
            cur_text_feature = cur_text_feature.unsqueeze(0).repeat(num_attn_img, 1, 1)

            hidden_states = torch.cat([cur_img_feature, cur_text_feature], dim=1) # [4, 642, 4096]

            attention_mask = None
            past_key_values_length = 0
            seq_length = hidden_states.shape[1]
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )

            position_ids = position_ids.unsqueeze(0).repeat(num_attn_img, 1)
            past_key_value = None

            # reuse llm
            with torch.no_grad():
                for _, decoder_layer in enumerate(llm[: self.resue_llm_layers]):
                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_value,
                        output_attentions=False,
                        use_cache=False,
                    )
                hidden_states = layer_outputs[0] # [4, 642, 4096]

            # 1. split text and image
            new_text_feature = hidden_states[:, cur_img_feature.shape[1]:, :]
            new_img_feature = hidden_states[:, :cur_img_feature.shape[1], :]

            new_text_feature = self.text_layers(new_text_feature) # [4, 66, 4096]
            new_img_feature = self.image_layers(new_img_feature) # [4, 576, 4096]

            # 2. compute simarity
            img_text_sim = torch.matmul(new_img_feature, new_text_feature.transpose(-2, -1)) # [4, 576, 66]

            # 3. compute weight between text and each attn_image
            combination_sim = []
            for i in range(num_attn_img):
                combination_sim.append(torch.mean(img_text_sim[i]))
            combination_sim = torch.stack(combination_sim) # e.g., [23.5781, 25.8750, 23.0000, 20.7031]

            # Training ranks lower-LLaVA-loss candidates first and optimizes their
            # selector scores to be larger, so inference must select the maximum
            # selector score as well.
            sorted_indices = torch.argsort(combination_sim, descending=True)
            # save_sorted_indices(sorted_indices)
            # print(sorted_indices[0])

            # 4. select topk, e.g., topk=1
            selected_img_feature.append(img_feature[sorted_indices[0]])

        # only 1 img/batch
        return selected_img_feature

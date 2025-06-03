#    Copyright 2024 Hao Zhang
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union, Dict
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

import transformers
from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

# from ...constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from longva.longva.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from transformers import Qwen2Config, Qwen2Model, Qwen2ForCausalLM
import pdb

# from .qwen.modeling_qwen import QWenLMHeadModel, QWenModel
# from .qwen.configuration_qwen import QWenConfig

class LlavaQwenConfig(Qwen2Config):
    model_type = "llava_qwen"


class LlavaQwenModel(LlavaMetaModel, Qwen2Model):
    config_class = LlavaQwenConfig

    def __init__(self, config: Qwen2Config):
        super(LlavaQwenModel, self).__init__(config)


class LlavaQwenForCausalLM(Qwen2ForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaQwenConfig

    def __init__(self, config):
        # super(Qwen2ForCausalLM, self).__init__(config)
        Qwen2ForCausalLM.__init__(self, config)
        config.model_type = "llava_qwen"
        config.rope_scaling = None

        self.model = LlavaQwenModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    # 4. 处理 prefix 部分
    def process_block(self, block_embeds, current_past_key_values=None, bsz=1, device=None, position_ids=None):

        if current_past_key_values is None:
            seq_len = block_embeds.size(1)
            position_ids = torch.arange(0, seq_len, device=device).expand(bsz, -1)
            attention_mask = torch.ones((bsz, seq_len), device=device, dtype=torch.long)
        else:
            seq_len = block_embeds.size(1)
            prefix_len = current_past_key_values[0][0].size(2)
            position_ids = torch.arange(prefix_len, prefix_len + seq_len, device=device).expand(bsz, -1)
            # position_ids = torch.arange(0, seq_len, device=device).expand(bsz, -1)
            attention_mask = torch.ones((bsz, prefix_len + seq_len), device=device, dtype=torch.long)
        

        outputs = self.model(
            inputs_embeds=block_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=current_past_key_values,
            use_cache=True,  # 虽然我们整体 use_cache=False，但这里要获取 KV
            return_dict=True,
        )
        return outputs.past_key_values

    def get_sparse_attention_mask(self, total_len, num_blocks, block_size, time_token_start_indices,  time_token_indices, visual_token_start_pos, visual_token_end_pos, attention_mask, inputs_embeds):

        causal_mask = torch.tril(torch.ones((total_len, total_len), dtype=torch.bool)).unsqueeze(0).repeat(1, 1, 1)
        mask = torch.zeros(total_len, total_len, dtype=torch.bool)
        start = visual_token_start_pos

        # last_start = None
        for i in range(num_blocks):
            next_time_token_pos = (i + 1)*block_size
            if next_time_token_pos >= len(time_token_start_indices):
                end = visual_token_end_pos
            else:
                end = visual_token_start_pos + time_token_start_indices[ next_time_token_pos ]

            mask[start:end, start:end] = True  # block 内允许访问
            # if last_start is not None:  # 可以看到上一个 block, roll cache 机制
            #     mask[start:end, last_start:start] = True
            # last_start = start
            start = end
            
            

        mask[:, :visual_token_start_pos] = True
        mask[visual_token_end_pos:, :] = True        

        # timestamp 所在位置全部能看到/被看到        
        for idx in time_token_indices:
            mask[visual_token_start_pos + idx, :] = True   # 整行设为 True
            mask[:, visual_token_start_pos + idx] = True   # 整列设为 True

        causal_mask = torch.tril(torch.ones(total_len, total_len, dtype=torch.bool))
        final_mask = (mask & causal_mask).unsqueeze(0).unsqueeze(0).to(dtype=attention_mask.dtype, device=attention_mask.device)

        # 假设 final_mask 是 [total_len, total_len]
        num_allowed = final_mask.sum().item()
        upper_triangle_num = total_len * (total_len + 1) // 2
        ratio = num_allowed / upper_triangle_num
        # print(f"实际 attention 计算量占 full attention 的 {ratio * 100:.2f}%")
        
        invert_mask = 1.0 - final_mask
        final_mask = ((1.0 - final_mask) * -1e9).to(dtype=inputs_embeds.dtype)
        # min_value = torch.finfo(torch.bfloat16).min
        # final_mask = final_mask.masked_fill_(invert_mask.bool(), min_value)
        # print(final_mask)
        return final_mask, ratio

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        modalities: Optional[List[str]] = ["image"],
        dpo_forward: Optional[bool] = False,
        cache_position=None,
        time_embedding=None,
        visual_token_start_pos=None,
        visual_token_end_pos=None,
        time_token_start_indices=None,
        frames_num=None,
        time_token_indices=None,
        time_token_end_indices=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        # decoding
        if input_ids is not None and input_ids.size(1) == 1:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        if inputs_embeds is None:
            (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels) = self.prepare_inputs_labels_for_multimodal(input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities, image_sizes,time_embedding)
        

        block_size = 36
        visual_token_start_pos = visual_token_start_pos
        visual_token_end_pos = visual_token_end_pos
        visual_len = visual_token_end_pos - visual_token_start_pos
        num_blocks = (frames_num  + block_size * 4 - 1) // (block_size * 4)
        print(f'block_size: {block_size}, num_blocks: {num_blocks}')
        
        full_inputs_embeds = inputs_embeds  # [bsz, seq_len, embed_dim]
        bsz, total_len, embed_dim = full_inputs_embeds.size()
        device = full_inputs_embeds.device

        prefix_embeds = full_inputs_embeds[:, :visual_token_start_pos, :]
        suffix_embeds = full_inputs_embeds[:, visual_token_end_pos:, :]
        visual_embeds = full_inputs_embeds[:, visual_token_start_pos:visual_token_end_pos, :]
        num_visual_tokens = visual_embeds.size(1)

        all_past_key_values = [[] for _ in range(len(self.model.layers))]  # 假设 model 有 layers 属性
        prefix_past_key_values = []  

        # 0. Process prefix
        if prefix_embeds.size(1) > 0:
            pkv = self.process_block(prefix_embeds, bsz=bsz, device=device)
            for i in range(len(pkv)):
                all_past_key_values[i].append(pkv[i])
                # setting2: w prefix embeds:
                prefix_past_key_values.append(pkv[i])

        # streaming inps
        # 首先获取每个 block 的位置信息
        blocks_positions = [[(0, 0, visual_token_start_pos)]] # 第一个 block 是system指令
        frames_groups = [(0, visual_token_start_pos)]
        for idx, (time_start, time_end) in enumerate(zip(time_token_start_indices, time_token_end_indices)):
            if idx + 1 < len(time_token_start_indices):
                frames_group_end = time_token_start_indices[idx + 1]
            else:
                frames_group_end = visual_token_end_pos
            frames_groups.append(
                (time_start, time_end, frames_group_end)
            )

        single_block = []
        for group in frames_groups[1:]: # 第一个system chunk直接跳过
            single_block.append(group)
            if len(single_block) == block_size:
                blocks_positions.append(single_block)
                single_block = []
        if len(single_block) != 0:
           blocks_positions.append(single_block)
        num_blocks = len(blocks_positions)


        block_streaming_past_key_values = prefix_past_key_values

        for idx, single_block in enumerate(blocks_positions[1:]):
            b_start, _, _ = single_block[0]
            _, _, b_end = single_block[0]
            visual_embeds_this_block = full_inputs_embeds[:,b_start:b_end,:]
            pkv = self.process_block(visual_embeds_this_block, current_past_key_values=block_streaming_past_key_values, bsz=bsz, device=device)
            for i in range(len(pkv)):
                key_this_block, val_this_block = pkv[i]
                all_past_key_values[i].append( pkv[i] )

                time_keys_list = []
                time_vals_list = []
                # 获取时间戳：
                for group in single_block:
                    time_start, time_end, _ = group
                    time_keys_list.append(key_this_block[:,time_start:time_end,:])
                    time_vals_list.append(val_this_block[:,time_start:time_end,:])
                
                time_keys = torch.cat(time_keys_list, dim=1)
                time_vals = torch.cat(time_vals_list, dim=1)
                past_k, past_v = block_streaming_past_key_values[i]
                block_streaming_past_key_values[i] = (torch.cat(past_k, time_keys, dim=1), torch.cat(past_v, time_keys, dim=1))


        # 2. 合并所有的 past kvs
        merged_pkv = []
        for layer_pkvs in all_past_key_values:
            if not layer_pkvs:
                continue
            keys = torch.cat([pkv[0] for pkv in layer_pkvs], dim=2)  # dim=2 是 sequence 维度
            values = torch.cat([pkv[1] for pkv in layer_pkvs], dim=2)
            merged_pkv.append((keys, values))

        # 3. Process suffix
        if suffix_embeds.size(1) > 0:
            seq_len = suffix_embeds.size(1)
            prefix_len = visual_token_end_pos
            position_ids = torch.arange(prefix_len, prefix_len + seq_len, device=device).expand(bsz, -1)
            # position_ids = torch.arange(0, seq_len, device=device).expand(bsz, -1)
            attention_mask = torch.ones((bsz, total_len), device=device, dtype=torch.long)

            outputs = super().forward(
                inputs_embeds=suffix_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=merged_pkv,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                use_cache=True,  # 虽然我们整体 use_cache=False，但这里要获取 KV
                return_dict=return_dict,
            )

        return outputs

    
    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        modalities: Optional[List[str]] = ["image"],
        time_embedding=None,
        gt_frame_idx=None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:

        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        # 获取位置信息 by qmh
        if images is not None and images[0].size(0) > 0:
            IMAGE_TOKEN_INDEX = -200
            TOKEN_PERFRAME = 36
            frames_num = images[0].size(0)
            visual_token_start_pos = (inputs == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[1].item()
            num_tokens = time_embedding[0].size(0)
            visual_token_end_pos = visual_token_start_pos + num_tokens
            kwargs['visual_token_start_pos'] = visual_token_start_pos
            kwargs['visual_token_end_pos'] = visual_token_end_pos
            # time_token_start_indices = (time_embedding[0] == 1462).nonzero(as_tuple=True)
            time_token_start_indices = (time_embedding[0] == 1462).nonzero(as_tuple=True)[0].cpu().tolist()
            kwargs['time_token_start_indices'] = [idx + visual_token_start_pos for idx in time_token_start_indices]
            # kwargs['time_token_start_indices'] = time_token_start_indices + visual_token_start_pos
            kwargs['frames_num'] = frames_num
            time_token_indices = (time_embedding[0] != 151654).nonzero(as_tuple=True)[0].cpu().tolist()
            kwargs['time_token_indices'] = [idx + visual_token_start_pos for idx in time_token_indices]
            time_token_end_indices = (time_embedding[0] == 25).nonzero(as_tuple=True)[0].cpu().tolist()
            kwargs['time_token_end_indices'] = [idx + visual_token_start_pos + 1 for idx in time_token_end_indices]
            # kwargs['time_token_end_indices'] = time_token_end_indices + visual_token_start_pos


        if images is not None:
            (inputs, position_ids, attention_mask, _, inputs_embeds, _) = self.prepare_inputs_labels_for_multimodal(inputs, position_ids, attention_mask, None, None, images, modalities, image_sizes=image_sizes,time_embedding=time_embedding)
        
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)
        
        return super().generate(position_ids=position_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, gt_frame_idx=gt_frame_idx, **kwargs)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, gt_frame_idx=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        visual_token_start_pos = kwargs.get("visual_token_start_pos", None)
        visual_token_end_pos = kwargs.get("visual_token_end_pos", None)
        time_token_start_indices = kwargs.get("time_token_start_indices", None)
        frames_num = kwargs.get("frames_num", None)
        time_token_indices = kwargs.get("time_token_indices", None)
        time_token_end_indices = kwargs.get("time_token_end_indices", None)

        inputs = super().prepare_inputs_for_generation(input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs)

        inputs["visual_token_start_pos"] = visual_token_start_pos
        inputs["visual_token_end_pos"] = visual_token_end_pos
        inputs["time_token_start_indices"] = time_token_start_indices
        inputs["frames_num"] = frames_num
        inputs["time_token_indices"] = time_token_indices
        inputs["time_token_end_indices"] = time_token_end_indices
        
        inputs['gt_frame_idx'] = gt_frame_idx

        if images is not None:
            inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        return inputs



AutoConfig.register("llava_qwen", LlavaQwenConfig)
AutoModelForCausalLM.register(LlavaQwenConfig, LlavaQwenForCausalLM)

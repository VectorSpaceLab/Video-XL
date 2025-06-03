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

    def get_sparse_attention_mask(self, total_len, num_blocks, block_size, time_token_start_indices, time_token_end_indices, time_token_indices, visual_token_start_pos, visual_token_end_pos, attention_mask, inputs_embeds, prev_blocks_num):

        causal_mask = torch.tril(torch.ones((total_len, total_len), dtype=torch.bool)).unsqueeze(0).repeat(1, 1, 1)
        mask = torch.zeros(total_len, total_len, dtype=torch.bool)
        start = visual_token_start_pos

        record_block_start = []
        for i in range(num_blocks):
            next_time_token_pos = (i + 1)*block_size
            if next_time_token_pos >= len(time_token_start_indices):
                end = visual_token_end_pos
            else:
                end =  time_token_start_indices[ next_time_token_pos ]

            mask[start:end, start:end] = True  # block 内允许访问

            if len(record_block_start) >= prev_blocks_num:
                prev_start = record_block_start[-prev_blocks_num]
            else:
                prev_start = visual_token_start_pos

            mask[start:end, prev_start:start] = True    # 当前 block 可以看到前 prev_blocks_num 个 block
            record_block_start.append(start)
            # current_timestamp_th = i*block_size # 当前Block开始的timestamp 是第几个时间戳
            # visiable_timestamps_start_indices = time_token_start_indices[max(current_timestamp_th-limit_visiable_timestamps, 0):current_timestamp_th]
            # visiable_timestamps_end_indices = time_token_end_indices[max(current_timestamp_th-limit_visiable_timestamps, 0):current_timestamp_th]

            # for time_start, time_end in zip(visiable_timestamps_start_indices, visiable_timestamps_end_indices):
            #     mask[start:end, time_start:time_end] = True
        
            start = end

            
        mask[:, :visual_token_start_pos] = True
        mask[visual_token_end_pos:, :] = True        
    
        # timestamp 所在位置全部能看到/被看到        
        for idx in time_token_indices:
            mask[idx, :] = True   # 整行设为 True
            mask[:, idx] = True   # 整列设为 True

        causal_mask = torch.tril(torch.ones(total_len, total_len, dtype=torch.bool))
        final_mask = (mask & causal_mask).unsqueeze(0).unsqueeze(0).to(dtype=attention_mask.dtype, device=attention_mask.device)

        # 假设 final_mask 是 [total_len, total_len]
        num_allowed = final_mask.sum().item()
        upper_triangle_num = total_len * (total_len + 1) // 2
        ratio = num_allowed / upper_triangle_num
        # print(f"实际 attention 计算量占 full attention 的 {ratio * 100:.2f}%")
        
        invert_mask = 1.0 - final_mask
        final_mask = ((1.0 - final_mask) * -1e9).to(dtype=inputs_embeds.dtype)
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
        time_token_end_indices=None,
        frames_num=None,
        time_token_indices=None,
        prev_blocks_num=None,
        block_size_chosed=None
    ) -> Union[Tuple, CausalLMOutputWithPast]:

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
        
        
        bsz, total_len, embed_dim = inputs_embeds.size()

        visual_token_start_pos = visual_token_start_pos
        visual_token_end_pos = visual_token_end_pos
        visual_len = visual_token_end_pos - visual_token_start_pos
        
        block_size_list = [2,4,8,16,32]
        best_block_size = None
        min_diff = float('inf')
       
        block_size = block_size_chosed
        num_blocks = (frames_num  + block_size * 4 - 1) // (block_size * 4)
        final_mask, ratio = self.get_sparse_attention_mask(total_len, num_blocks, block_size, time_token_start_indices, time_token_end_indices, time_token_indices, visual_token_start_pos, visual_token_end_pos, attention_mask, inputs_embeds, prev_blocks_num)

        print(f'frames:{frames_num}, block_num:{num_blocks}, bsz:{block_size}, prev_blocks_num:{prev_blocks_num}, ratio:{ratio}') 

        return super().forward(
            input_ids=input_ids,
            attention_mask=final_mask, # final_mask
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        modalities: Optional[List[str]] = ["image"],
        time_embedding=None,
        prev_blocks_num=None,
        block_size_chosed=None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:

        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        kwargs['prev_blocks_num'] = prev_blocks_num
        kwargs['block_size_chosed'] = block_size_chosed
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

            kwargs['frames_num'] = frames_num
            time_token_indices = (time_embedding[0] != 151654).nonzero(as_tuple=True)[0].cpu().tolist()
            kwargs['time_token_indices'] = [idx + visual_token_start_pos for idx in time_token_indices]

            time_token_end_indices = (time_embedding[0] == 25).nonzero(as_tuple=True)[0].cpu().tolist()
            kwargs['time_token_end_indices'] = [idx + visual_token_start_pos + 1 for idx in time_token_end_indices]
            
        #print(images[0].shape)
        if images is not None:
            (inputs, position_ids, attention_mask, _, inputs_embeds, _) = self.prepare_inputs_labels_for_multimodal(inputs, position_ids, attention_mask, None, None, images, modalities, image_sizes=image_sizes,time_embedding=time_embedding)
        
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)
        
        #print(inputs_embeds.shape)
        return super().generate(position_ids=position_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
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
        inputs["time_token_end_indices"] = time_token_end_indices
        inputs["frames_num"] = frames_num
        inputs["time_token_indices"] = time_token_indices
        
        inputs["prev_blocks_num"] = kwargs.get("prev_blocks_num", None)
        inputs["block_size_chosed"] = kwargs.get("block_size_chosed", None)

        if images is not None:
            inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        return inputs


AutoConfig.register("llava_qwen", LlavaQwenConfig)
AutoModelForCausalLM.register(LlavaQwenConfig, LlavaQwenConfig)

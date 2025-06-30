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
# from transformers import Qwen2Config, Qwen2Model, Qwen2ForCausalLM
from .modeling_qwen2 import Qwen2Config, Qwen2Model, Qwen2ForCausalLM
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
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        # if inputs_embeds is None:
        #     (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels) = self.prepare_inputs_labels_for_multimodal(input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities, image_sizes,time_embedding)
        
        # print("input_ids",input_ids)
        # print("input_embeds",inputs_embeds.shape)
        # print("labels",labels)
        # print("mask",attention_mask)

        # if dpo_forward:
        #     outputs = self.model(
        #         input_ids=input_ids,
        #         attention_mask=attention_mask,
        #         position_ids=position_ids,
        #         past_key_values=past_key_values,
        #         inputs_embeds=inputs_embeds,
        #         use_cache=use_cache,
        #         output_attentions=output_attentions,
        #         output_hidden_states=output_hidden_states,
        #         return_dict=return_dict,
        #     )

        #     hidden_states = outputs[0]
        #     logits = self.lm_head(hidden_states)
        #     return logits, labels

        # else:
        #     return super().forward(
        #         input_ids=input_ids,
        #         attention_mask=attention_mask,
        #         position_ids=position_ids,
        #         past_key_values=past_key_values,
        #         inputs_embeds=inputs_embeds,
        #         labels=labels,
        #         use_cache=use_cache,
        #         output_attentions=output_attentions,
        #         output_hidden_states=output_hidden_states,
        #         return_dict=return_dict,
        #     )

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

        # 0. 先提前获取开头和结尾的索引
        block_size = 360
        print(f'block_size: {block_size}')
        overlap = 72 # setting3 overlap 传递消息，或者用文本传递消息, "summarize the above video content."
        # visual_token_start_pos = 14
        # num_tokens = 36 * 150
        # visual_token_end_pos = visual_token_start_pos  + num_tokens

        full_inputs_embeds = inputs_embeds  # [bsz, seq_len, embed_dim]
        bsz, total_len, embed_dim = full_inputs_embeds.size()
        device = full_inputs_embeds.device

        # 1. 提取 prefix、visual_tokens 和 suffix
        prefix_embeds = full_inputs_embeds[:, :visual_token_start_pos, :]
        suffix_embeds = full_inputs_embeds[:, visual_token_end_pos:, :]
        visual_embeds = full_inputs_embeds[:, visual_token_start_pos:visual_token_end_pos, :]
        num_visual_tokens = visual_embeds.size(1)

        # 2. 拆分视觉 token 成多个 block
        blocks = []
        for start in range(0, num_visual_tokens, block_size): 
        # setting3: overlap
        # for start in range(0, num_visual_tokens, block_size - overlap):
            end = min(start + block_size, num_visual_tokens)
            block = visual_embeds[:, start:end, :]
            blocks.append(block)

        # 3. 初始化保存所有块的 past_key_values
        all_past_key_values = [[] for _ in range(len(self.model.layers))]  # 假设 model 有 layers 属性
        prefix_past_key_values = []  

        # 5. 逐个处理 prefix、每个视觉 block、suffix，并收集各自的 past_key_values
        # 注意：每个块都是独立 forward，不依赖前面的 KV（因为我们没有传递 past_key_values）
        
        # Process prefix
        if prefix_embeds.size(1) > 0:
            pkv = self.process_block(prefix_embeds, bsz=bsz, device=device)
            for i in range(len(pkv)):
                all_past_key_values[i].append(pkv[i])
                # setting2: w prefix embeds:
                prefix_past_key_values.append(pkv[i])

        # pdb.set_trace()
        # Process visual blocks
        for idx, block in enumerate(blocks):
            # Exp2 仅改变位置编码
            # pkv = self.process_block(block, prefix_past_key_values, bsz=bsz, device=device)
            # for i in range(len(pkv)):
            #     prefix_past_key_values[i] = pkv[i]
            
            # setting1: no prefix embeds
            pkv = self.process_block(block, bsz=bsz, device=device)
            for i in range(len(pkv)):
                all_past_key_values[i].append(pkv[i])

            # setting2: w prefix embeds:
            # pkv = self.process_block(block, prefix_past_key_values, bsz=bsz, device=device)
            # for i in range(len(pkv)):
            #     filter_pkv = (pkv[i][0][:,:,visual_token_start_pos:,:], pkv[i][1][:,:,visual_token_start_pos:,:])
            #     all_past_key_values[i].append(filter_pkv)

            # setting3: overlap
            # pkv = self.process_block(block, bsz=bsz, device=device)
            # for i in range(len(pkv)):
            #     if idx == 0:
            #         all_past_key_values[i].append(pkv[i])
            #     else:
            #         filter_pkv = (pkv[i][0][:,:,overlap:,:], pkv[i][1][:,:,overlap:,:])
            #         all_past_key_values[i].append(filter_pkv)


        merged_pkv = []
        for layer_pkvs in all_past_key_values:
            if not layer_pkvs:
                continue
            keys = torch.cat([pkv[0] for pkv in layer_pkvs], dim=2)  # dim=2 是 sequence 维度
            values = torch.cat([pkv[1] for pkv in layer_pkvs], dim=2)
            merged_pkv.append((keys, values))

        # Exp2 仅改变编码位置
        # merged_pkv = prefix_past_key_values

        # Process suffix
        if suffix_embeds.size(1) > 0:
            seq_len = suffix_embeds.size(1)
            prefix_len = visual_token_end_pos
            position_ids = torch.arange(prefix_len, prefix_len + seq_len, device=device).expand(bsz, -1)
            # position_ids = torch.arange(0, seq_len, device=device).expand(bsz, -1)
            attention_mask = torch.ones((bsz, total_len), device=device, dtype=torch.long)
            # print(f'suffix_embeds shape: {suffix_embeds.shape}')
            # print(f'position_ids shape: {position_ids.shape}')
            # print(f'position_ids start: {position_ids[:,:3]}')
            # print(f'attention_mask shape: {attention_mask.shape}')
            # print(f'merged_pkv shape: {merged_pkv[0][0].shape}')
            # print(f'='*50)
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
            num_tokens = TOKEN_PERFRAME * frames_num
            visual_token_end_pos = visual_token_start_pos + num_tokens
            kwargs['visual_token_start_pos'] = visual_token_start_pos
            kwargs['visual_token_end_pos'] = visual_token_end_pos
            
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

        inputs = super().prepare_inputs_for_generation(input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs)

        inputs["visual_token_start_pos"] = visual_token_start_pos
        inputs["visual_token_end_pos"] = visual_token_end_pos

        if images is not None:
            inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        return inputs


AutoConfig.register("llava_qwen", LlavaQwenConfig)
AutoModelForCausalLM.register(LlavaQwenConfig, LlavaQwenForCausalLM)

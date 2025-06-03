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
from longva.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from transformers import Qwen2Config, Qwen2Model, Qwen2ForCausalLM
import random
import pdb
import torch.distributed as dist
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


    def get_best_prev_blocks_num_img(self, frames_num):
        # block size 4, 用于300帧
        if frames_num < 20:
            return 0
        elif frames_num < 30:
            return 1
        elif frames_num < 60:
            return 3
        elif frames_num < 90:
            return 5
        elif frames_num < 130:
            return 7
        else:
            return 9
    def get_best_prev_blocks_num(self, frames_num):
        # block size 4, 用于300帧
        if frames_num < 60:
            return 0
        elif frames_num < 120:
            return 1
        elif frames_num < 200:
            return 2
        else:
            return 3

        # block size 2
        # if frames_num < 60:
        #     return 1
        # elif frames_num < 100:
        #     return 2
        # elif frames_num < 130:
        #     return 3
        # else:
        #     return 4


        # for frames in range(30, max_frames+10, 10):
        #     group = 4 # 每个group 4帧
        #     block_size = 4 # 每个 block 有 2 个 group
        #     block_num = (frames + block_size*group - 1) // (block_size*group)
        #     block_tokens = 144*block_size
        #     prev_nums = 0
        #     print(f'frames: {frames}, block_size: {block_size}, block_num: {block_num}, block_tokens: {block_tokens}, prev_nums: {prev_nums}')

        #     full_attn_cal = (block_num*block_tokens)**2//2

        #     spart_attn_cal = 0
        #     for i in range(1, block_num+1):
        #         if i <= prev_nums:
        #             spart_attn_cal += block_tokens**2//2 + (i-1) * block_tokens**2
        #         else:
        #             spart_attn_cal += block_tokens**2//2 + prev_nums * block_tokens**2

        #     print(f'flops preservation: {spart_attn_cal/full_attn_cal*100:.2f}')


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
        path=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        IMAGE_TOKEN_INDEX = -200
        try:
            visual_token_start_pos = (input_ids == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[1].item()
        except:
            print('no image token')
            print(path)
            print(input_ids)
            print(type(images))
            print(images.size())
            exit()
        if images is not None and images[0].size(0) > 0 and time_embedding is not None and time_embedding[0] is not None:
            frames_num = images[0].size(0)
            num_tokens = time_embedding[0].size(0)
            visual_token_end_pos = visual_token_start_pos + num_tokens
            time_token_start_indices = (time_embedding[0] == 1462).nonzero(as_tuple=True)[0].cpu().tolist()
            time_token_indices = (time_embedding[0] != 151654).nonzero(as_tuple=True)[0].cpu().tolist()

        if inputs_embeds is None:
            (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels) = self.prepare_inputs_labels_for_multimodal(input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities, image_sizes,time_embedding)

        if time_embedding is None or time_embedding[0] is None: # 说明是图像数据
            is_image = True
        else:
            is_image = False

        bsz, total_len, embed_dim = inputs_embeds.size()
        
        if is_image:
            IMAGE_TOKEN_INDEX = -200
            tokens_per_patch = 144
            block_size = 1
            patch_num = images.size(1)  # 每个 patch 实际上被 repeat 四次，
            equalvant_frame_num = patch_num * 4

            # num_blocks = (patch_num  + block_size  - 1) // (block_size)
            num_blocks = (equalvant_frame_num  + block_size * 4 - 1) // (block_size * 4)

            visual_len = patch_num * tokens_per_patch
            visual_token_end_pos = visual_token_start_pos + visual_len
            mask = torch.zeros(total_len, total_len, dtype=torch.bool)
           
            prev_blocks_num = self.get_best_prev_blocks_num_img(equalvant_frame_num)
            record_block_start = []
            start = visual_token_start_pos

            for i in range(num_blocks):
                end = min(start + block_size * tokens_per_patch, visual_token_end_pos)
                mask[start:end, start:end] = True  # block 内允许访问

                if prev_blocks_num != 0:
                    if len(record_block_start) >= prev_blocks_num:
                        prev_start = record_block_start[-prev_blocks_num] # mask[start-3:end+2, prev_start-1:start]
                    else:
                        prev_start = visual_token_start_pos

                    mask[start:end, prev_start:start] = True    # 当前 block 可以看到前 prev_blocks_num 个 block

                record_block_start.append(start)
                start = end
                
            mask[:, :visual_token_start_pos] = True
            mask[visual_token_end_pos:, :] = True                    

        else:
            block_size = 4 # 8帧
            num_blocks = (frames_num  + block_size * 4 - 1) // (block_size * 4)
            visual_len = visual_token_end_pos - visual_token_start_pos

            mask = torch.zeros(total_len, total_len, dtype=torch.bool)
            start = visual_token_start_pos

            # prev_blocks_num = 7
            # 为不同长度的视频自动确定 prev_blocks_num, 
            prev_blocks_num = self.get_best_prev_blocks_num(frames_num)

            record_block_start = []
            for i in range(num_blocks):
                # if time_embedding is None or time_embedding[0] is None:
                next_time_token_pos = (i + 1)*block_size
                if next_time_token_pos >= len(time_token_start_indices):
                    end = visual_token_end_pos
                else:
                    end = visual_token_start_pos + time_token_start_indices[ next_time_token_pos ]

                mask[start:end, start:end] = True  # block 内允许访问

                if prev_blocks_num != 0:
                    if len(record_block_start) >= prev_blocks_num:
                        prev_start = record_block_start[-prev_blocks_num] # mask[start-3:end+2, prev_start-1:start]
                    else:
                        prev_start = visual_token_start_pos

                    mask[start:end, prev_start:start] = True    # 当前 block 可以看到前 prev_blocks_num 个 block

                record_block_start.append(start)
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

        if is_image:
            print(f"f'Image !! patch num: {patch_num}, block_size: {1}, num_blocks: {num_blocks}, prev_blocks_num:{prev_blocks_num}, total_len:{total_len}, preserve {ratio * 100:.2f}%")
        else:
            print(f"f'Video !! frames num: {frames_num}, block_size: {block_size}, prev_blocks_num:{prev_blocks_num}, total_len:{total_len}, preserve {ratio * 100:.2f}%")

        invert_mask = ~final_mask
        final_mask = (invert_mask * -1e9).to(dtype=inputs_embeds.dtype)

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
            kwargs['time_token_start_indices'] = time_token_start_indices
            kwargs['frames_num'] = frames_num
            time_token_indices = (time_embedding[0] != 151654).nonzero(as_tuple=True)[0].cpu().tolist()
            kwargs['time_token_indices'] = time_token_indices
            
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

        inputs = super().prepare_inputs_for_generation(input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs)

        inputs["visual_token_start_pos"] = visual_token_start_pos
        inputs["visual_token_end_pos"] = visual_token_end_pos
        inputs["time_token_start_indices"] = time_token_start_indices
        inputs["frames_num"] = frames_num
        inputs["time_token_indices"] = time_token_indices

        if images is not None:
            inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        return inputs


AutoConfig.register("llava_qwen", LlavaQwenConfig)
AutoModelForCausalLM.register(LlavaQwenConfig, LlavaQwenForCausalLM)



# class LlavaQwenForCausalLM(Qwen2ForCausalLM, LlavaMetaForCausalLM):
#     config_class = LlavaQwenConfig

#     def __init__(self, config):
#         # super(Qwen2ForCausalLM, self).__init__(config)
#         Qwen2ForCausalLM.__init__(self, config)
#         config.model_type = "llava_qwen"
#         config.rope_scaling = None

#         self.model = LlavaQwenModel(config)
#         self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
#         # Initialize weights and apply final processing
#         self.post_init()

#     def get_model(self):
#         return self.model

#     def forward(
#         self,
#         input_ids: torch.LongTensor = None,
#         attention_mask: Optional[torch.Tensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         past_key_values: Optional[List[torch.FloatTensor]] = None,
#         inputs_embeds: Optional[torch.FloatTensor] = None,
#         labels: Optional[torch.LongTensor] = None,
#         use_cache: Optional[bool] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         images: Optional[torch.FloatTensor] = None,
#         image_sizes: Optional[List[List[int]]] = None,
#         return_dict: Optional[bool] = None,
#         modalities: Optional[List[str]] = ["image"],
#         dpo_forward: Optional[bool] = False,
#         cache_position=None,
#         time_embedding=None,
#         path=None,
#     ) -> Union[Tuple, CausalLMOutputWithPast]:
#         # print(f'path:{path}')

#         if inputs_embeds is None:
#             (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels) = self.prepare_inputs_labels_for_multimodal(input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities, image_sizes,time_embedding)
        
#         # print("input_ids",input_ids)
#         # print("input_embeds",inputs_embeds.shape)
#         # if labels is not None:
#         #     num_neg_ones = (labels < 0).sum().item()
#         #     print(f"-1 in labels: {num_neg_ones}")
#         # if inputs_embeds.shape[1]>4096:
#         #     inputs_embeds=inputs_embeds[:,:4096,:]
#         # print(inputs_embeds.shape)
#         # print("labels",labels)
#         # print("mask",attention_mask)

#         if dpo_forward:
#             outputs = self.model(
#                 input_ids=input_ids,
#                 attention_mask=attention_mask,
#                 position_ids=position_ids,
#                 past_key_values=past_key_values,
#                 inputs_embeds=inputs_embeds,
#                 use_cache=use_cache,
#                 output_attentions=output_attentions,
#                 output_hidden_states=output_hidden_states,
#                 return_dict=return_dict,
#             )

#             hidden_states = outputs[0]
#             logits = self.lm_head(hidden_states)
#             return logits, labels

#         else:
#             return super().forward(
#                 input_ids=input_ids,
#                 attention_mask=attention_mask,
#                 position_ids=position_ids,
#                 past_key_values=past_key_values,
#                 inputs_embeds=inputs_embeds,
#                 labels=labels,
#                 use_cache=use_cache,
#                 output_attentions=output_attentions,
#                 output_hidden_states=output_hidden_states,
#                 return_dict=return_dict,
#             )

#     @torch.no_grad()
#     def generate(
#         self,
#         inputs: Optional[torch.Tensor] = None,
#         images: Optional[torch.Tensor] = None,
#         image_sizes: Optional[torch.Tensor] = None,
#         modalities: Optional[List[str]] = ["image"],
#         **kwargs,
#     ) -> Union[GenerateOutput, torch.LongTensor]:
#         position_ids = kwargs.pop("position_ids", None)
#         attention_mask = kwargs.pop("attention_mask", None)
#         if "inputs_embeds" in kwargs:
#             raise NotImplementedError("`inputs_embeds` is not supported")

#         if images is not None:
#             (inputs, position_ids, attention_mask, _, inputs_embeds, _) = self.prepare_inputs_labels_for_multimodal(inputs, position_ids, attention_mask, None, None, images, modalities, image_sizes=image_sizes)
#         else:
#             inputs_embeds = self.get_model().embed_tokens(inputs)

#         return super().generate(position_ids=position_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs)

#     def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
#         images = kwargs.pop("images", None)
#         image_sizes = kwargs.pop("image_sizes", None)
#         inputs = super().prepare_inputs_for_generation(input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs)
#         if images is not None:
#             inputs["images"] = images
#         if image_sizes is not None:
#             inputs["image_sizes"] = image_sizes
#         return inputs


# AutoConfig.register("llava_qwen", LlavaQwenConfig)
# AutoModelForCausalLM.register(LlavaQwenConfig, LlavaQwenForCausalLM)

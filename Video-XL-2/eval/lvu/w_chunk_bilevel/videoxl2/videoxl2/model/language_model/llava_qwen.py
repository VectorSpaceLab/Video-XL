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
from videoxl2.videoxl2.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from .modeling_qwen2 import Qwen2Config, Qwen2Model, Qwen2ForCausalLM
import pdb
import time
# from .qwen.modeling_qwen import QWenLMHeadModel, QWenModel
# from .qwen.configuration_qwen import QWenConfig
import random
random.seed(42)
import torch
from statistics import mean
import torch.nn.functional as F

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

    def uniform_sampling(self, embeds, start_idx, end_idx, step):
        indices = torch.arange(start_idx, end_idx, step).to(device=embeds.device)
        return embeds.index_select(1, indices), indices
    def pooling_sampling(self, embeds, start_idx, end_idx, step, pool_type='avg'):
        selected = embeds[:, start_idx:end_idx, :]
        B, D, L = selected.shape
        kernel_size = step
        stride = step
        
        selected_transposed = selected.transpose(1, 2)  # shape: (1, 12, 4)

        if pool_type == 'avg_pool':
            pooled = F.avg_pool1d(selected_transposed, kernel_size=kernel_size, stride=stride)
        elif pool_type == 'max_pool':
            pooled = F.max_pool1d(selected_transposed, kernel_size=kernel_size, stride=stride)
        else:
            raise ValueError(f"Unsupported pooling type: {pool_type}")

        pooled = pooled.transpose(1, 2)  # shape: (1, 2, 12)
        return pooled, torch.arange(start_idx, start_idx + pooled.shape[1] * step, step).to(device=embeds.device)

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

    def pooling_kvs(self, kvs, step):
        # kvs shape: (bsz, 4, seq_len, head_dim)
        kernel_size = step
        stride = step
        # kvs = kvs.transpose(2, 3)
        # pooled_kvs = F.avg_pool1d(kvs, kernel_size=kernel_size, stride=stride)

        # Option 2: 模拟滑动窗口的平均池化
        # 这需要手动实现滑动窗口或借助 `unfold`
        # 更简单的方法是使用 `F.avg_pool1d` 配合 `view/permute`，如之前所示
        # 或者，如果你想对每个 "head" 的每个 "feature" 维度进行独立的一维池化，
        # 并且 `128` 是你的序列长度，`144` 是通道/特征维度，你可以：
        kvs_permuted = kvs.permute(0, 1, 3, 2) # (batch_size, num_heads, feature_dim, sequence_length)
        # 然后展平 batch_size 和 num_heads
        N_flat = kvs_permuted.shape[0] * kvs_permuted.shape[1]
        C = kvs_permuted.shape[2]
        L = kvs_permuted.shape[3]
        kvs_for_pool = kvs_permuted.reshape(N_flat, C, L)
        pooled_kvs = F.avg_pool1d(kvs_for_pool, kernel_size=kernel_size, stride=stride)
        # 再恢复形状
        pooled_kvs_restored = pooled_kvs.view(kvs.shape[0], kvs.shape[1], pooled_kvs.shape[1], pooled_kvs.shape[2]).permute(0, 1, 3, 2)
        return pooled_kvs_restored


    def get_sparse_attention_mask(self, total_len, num_blocks, block_size, time_token_start_indices, time_token_end_indices, time_token_indices, visual_token_start_pos, visual_token_end_pos, attention_mask, inputs_embeds, prev_blocks_num=None):

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
        frames_num=None,
        time_token_indices=None,
        time_token_end_indices=None,
        prev_blocks_num=None,
        block_size_chosed=None,
        selected_unit_indices=None,
        selected_config=None
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if input_ids is not None and input_ids.size(1) == 1:
            past_key_len = past_key_values[0][0].size(-2)
            if position_ids[0][0] != past_key_len:
                position_ids = torch.tensor([[past_key_len]]).to(device=position_ids.device, dtype=position_ids.dtype)

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

        if selected_unit_indices is None:
            print(f'some not have selected_unit_indices')
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
        min_diff = float('inf')
        
        prefill_block_size = block_size_chosed
        prefill_num_blocks = (frames_num  + prefill_block_size * 4 - 1) // (prefill_block_size * 4)

        final_mask, ratio = self.get_sparse_attention_mask(total_len, prefill_num_blocks, prefill_block_size, time_token_start_indices, time_token_end_indices, time_token_indices, visual_token_start_pos, visual_token_end_pos, attention_mask, inputs_embeds, prev_blocks_num=prev_blocks_num)

        # print(f'frames:{frames_num}, block_num:{prefill_num_blocks}, bsz:{prefill_block_size},prev_blocks_num:{prev_blocks_num}, ratio:{ratio}') 

        outputs = super().forward(
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

        if selected_unit_indices is None:
            print(f'Directly return outputs')
            return outputs

        #############################
        # bi-level KVs Decoding
        pkv = outputs.past_key_values
        topk = selected_config['selected_topk']
        block_size = selected_config['selected_chunk_size']
        step = selected_config['compression'] # 144/12 = 12

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

        # 获取压缩 KV 表征
        cmpr_kvs = [ [] for i in range(len(pkv)) ]
        for idx, block_infos in enumerate(blocks_positions):
            for i in range(len(pkv)):
                pkv_thislayer = pkv[i]
                if idx == 0:
                    cmpr_kvs[i].append( (pkv_thislayer[0][:,:,0:visual_token_start_pos], pkv_thislayer[1][:,:,0:visual_token_start_pos]) )
                    continue

                cmpr_keys_this_block = []
                cmpr_vals_this_block = []

                for time_start, time_end, group_end in block_infos:
                    time_keys = pkv_thislayer[0][:, :, time_start: time_end].clone()
                    time_vals = pkv_thislayer[1][:, :, time_start: time_end].clone()

                    visual_keys = pkv_thislayer[0][:, :, time_end:group_end].clone()
                    visual_vals = pkv_thislayer[1][:, :, time_end:group_end].clone()

                    pooled_visual_keys = self.pooling_kvs(visual_keys, step)
                    pooled_visual_values = self.pooling_kvs(visual_vals, step)

                    cmpr_keys_this_block.append(time_keys)
                    cmpr_keys_this_block.append(pooled_visual_keys)

                    cmpr_vals_this_block.append(time_vals)
                    cmpr_vals_this_block.append(pooled_visual_values)
                
                cmpr_kvs[i].append( (torch.cat(cmpr_keys_this_block, dim=2), torch.cat(cmpr_vals_this_block, dim=2) )  )
        
        
        # 使用 gt group
        gt_block_idx = []
        for this_frame_idx in selected_unit_indices:   # 按照八帧为单位生成的。
            block_idx = this_frame_idx//(block_size/2) + 1
            if block_idx not in gt_block_idx:
                gt_block_idx.append(block_idx)

        # make selected block indices
        if len(gt_block_idx) > 0: 
            indices = [0,1,len(blocks_positions)-1] + gt_block_idx
            indices = list(set(indices))
        else:
            indices = 'all'
        
        print(f'seleted indices: {indices}')
        full_inputs_embeds = inputs_embeds  # [bsz, seq_len, embed_dim]
        bsz, total_len, embed_dim = full_inputs_embeds.size()
        device = full_inputs_embeds.device
        suffix_embeds = full_inputs_embeds[:, visual_token_end_pos:, :]
       
        # 再根据索引取出 block，确保顺序不变
        mixed_prefill_past_key_values = []
        for i in range(len(pkv)):   # 遍历每一层
            merge_past_key = []
            merge_past_val = []            
            full_kvs_thislayer = pkv[i]
            cmpr_kvs_thislayer = cmpr_kvs[i]

            for block_idx, single_block in enumerate(blocks_positions):
                if indices=='all' or block_idx in indices:
                    past_key, past_val = full_kvs_thislayer
                    f_time_start, f_time_end, f_frames_group_end = single_block[0]
                    l_time_start, l_time_end, l_frames_group_end = single_block[-1]
                    start = f_time_start
                    end = l_frames_group_end
                    merge_past_key.append(past_key[:,:,start:end].clone())
                    merge_past_val.append(past_val[:,:,start:end].clone())    
                else: 
                    past_key, past_val = cmpr_kvs_thislayer[block_idx]
                    merge_past_key.append(past_key.clone())
                    merge_past_val.append(past_val.clone())        
                
            mixed_past_key = torch.cat(merge_past_key, dim=2)
            mixed_past_val = torch.cat(merge_past_val, dim=2)
            mixed_prefill_past_key_values.append( (mixed_past_key, mixed_past_val) )

        prefill_len = mixed_past_key.size(2)
        total_len = prefill_len + suffix_embeds.size(1)

        print(f'{{"frames":{frames_num}, "prefill chunk size":{prefill_block_size}, "prefill chunks":{prefill_num_blocks}, "pre-filling flops reduction":{100-ratio*100:.2f}%, "bi-level kvs len":{prefill_len}, "original kvs len": {visual_token_end_pos}, "text query len":{suffix_embeds.size(1)}, "decoding kv reduction": {100-(prefill_len/visual_token_end_pos*100):.2f}%}}')

        # Process suffix
        if suffix_embeds.size(1) > 0:
            seq_len = suffix_embeds.size(1)
            # prefill_len = visual_token_end_pos
            position_ids = torch.arange(prefill_len, prefill_len + seq_len, device=device).expand(bsz, -1)
            attention_mask = torch.ones((bsz, total_len), device=device, dtype=torch.long)

            outputs = super().forward(
                inputs_embeds=suffix_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=mixed_prefill_past_key_values,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                use_cache=True,
                return_dict=return_dict,
                # blocks_positions=None,
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
        selected_unit_indices=None,
        prev_blocks_num=None,
        block_size_chosed=None,
        selected_config=None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:

        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)

        kwargs['prev_blocks_num'] = prev_blocks_num
        kwargs['block_size_chosed'] = block_size_chosed
        kwargs['selected_config'] = selected_config

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

        #print(images[0].shape)
        if images is not None:
            (inputs, position_ids, attention_mask, _, inputs_embeds, _) = self.prepare_inputs_labels_for_multimodal(inputs, position_ids, attention_mask, None, None, images, modalities, image_sizes=image_sizes,time_embedding=time_embedding)
        
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        #print(inputs_embeds.shape)
        return super().generate(position_ids=position_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, selected_unit_indices=selected_unit_indices, **kwargs)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, selected_unit_indices=None, **kwargs):
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

        inputs['selected_unit_indices'] = selected_unit_indices

        inputs["prev_blocks_num"] = kwargs.get("prev_blocks_num", None)
        inputs["block_size_chosed"] = kwargs.get("block_size_chosed", None)
        inputs["selected_config"] = kwargs.get("selected_config", None)

        if images is not None:
            inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        return inputs


AutoConfig.register("llava_qwen", LlavaQwenConfig)
AutoModelForCausalLM.register(LlavaQwenConfig, LlavaQwenConfig)

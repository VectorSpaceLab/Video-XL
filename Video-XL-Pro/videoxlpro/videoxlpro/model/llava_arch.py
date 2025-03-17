from abc import ABC, abstractmethod

import math
import re
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from .multimodal_encoder.builder import build_vision_tower
from .multimodal_resampler.builder import build_vision_resampler
from .multimodal_projector.builder import build_vision_projector
from transformers import AutoTokenizer

from videoxlpro.videoxlpro.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from videoxlpro.videoxlpro.mm_utils import get_anyres_image_grid_shape
from videoxlpro.videoxlpro.utils import rank0_print
import random
from .sae import SiglipAE

import torch.nn.functional as F

class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            delay_load = getattr(config, "delay_load", False)
            self.vision_tower = build_vision_tower(config, delay_load=delay_load)
            self.vision_resampler = build_vision_resampler(config, vision_tower=self.vision_tower)
            self.mm_projector = build_vision_projector(config, vision_cfg=self.vision_tower.config)

            if "unpad" in getattr(config, "mm_patch_merge_type", ""):
                self.image_newline = nn.Parameter(torch.empty(config.hidden_size, dtype=self.dtype))
    
        # self.llm_tokenizer = AutoTokenizer.from_pretrained(config._name_or_path)
        self.hidden_size=config.hidden_size
        # print(config)
        # exit(0)
        
#         self.text_tokenizer = T5Tokenizer.from_pretrained('google-t5/t5-small')
##############################################################################
#         self.text_select_model = T5EncoderModel.from_pretrained('google-t5/t5-small')
        
#         self.text_gamma=0.75
        
#         self.text_mlp=nn.Sequential(
#             nn.Linear(512,config.hidden_size),
#             nn.GELU(),
#         )
###############################################################################
        self.text_mlp=nn.Sequential(
            nn.Linear(config.hidden_size,config.hidden_size),
            nn.GELU(),
        )
        self.sae=SiglipAE()
        #self.sae.load_state_dict(torch.load('/share/LXRlxr0_0/code/videoxlturbo2.0/lmms-eval/longva/longva/model/encoder.pth'),strict=False)
        
###############################################################################
        # self.vision_select=nn.Parameter(
        #         torch.randn((4, self.config.hidden_size), dtype=self.dtype)
        # )
##############################################################################
        
    def get_vision_tower(self):
        vision_tower = getattr(self, "vision_tower", None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type

        self.config.mm_vision_tower = vision_tower
        self.config.vision_tower_pretrained = getattr(model_args, "vision_tower_pretrained", "")

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)
            vision_resampler = build_vision_resampler(model_args, vision_tower=vision_tower)
            for k, v in vision_resampler.config.items():
                setattr(self.config, k, v)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
                self.vision_resampler = [vision_resampler]
            else:
                self.vision_tower = vision_tower
                self.vision_resampler = vision_resampler
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_resampler = self.vision_resampler[0]
                vision_tower = self.vision_tower[0]
            else:
                vision_resampler = self.vision_resampler
                vision_tower = self.vision_tower
            vision_tower.load_model()

            # In case it is frozen by LoRA
            for p in self.vision_resampler.parameters():
                p.requires_grad = True

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, "mm_projector_type", "linear")
        self.config.mm_hidden_size = getattr(vision_resampler, "hidden_size", vision_tower.hidden_size)
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type
        
        self.sae=SiglipAE()
        self.sae.load_state_dict(torch.load('/share/LXRlxr0_0/code/videoxlturbo2.0/lmms-eval/longva/longva/model/encoder.pth'),strict=False)
        ##############################################################################
#         self.vision_select=nn.Parameter(
#                 torch.randn((30, self.config.hidden_size), dtype=self.dtype)
#         )
        
#         #self.text_tokenizer = T5Tokenizer.from_pretrained('google-t5/t5-small')
#         self.text_select_model = T5EncoderModel.from_pretrained('google-t5/t5-small')
        
#         self.text_mlp=nn.Sequential(
#             nn.Linear(512,self.config.hidden_size),
#             nn.GELU(),
#             # nn.Linear(config.hidden_size,config.hidden_size),
#             # nn.GELU(),
#         )
        ##############################################################################
        
        
        if getattr(self, "mm_projector", None) is None:
            self.mm_projector = build_vision_projector(self.config, vision_cfg=vision_tower.config)

            if "unpad" in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std)
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location="cpu")

            def get_w(weights, keyword):
                return {k.split(keyword + ".")[1]: v for k, v in weights.items() if keyword in k}

            incompatible_keys = self.mm_projector.load_state_dict(get_w(mm_projector_weights, "mm_projector"))
            rank0_print(f"Loaded mm projector weights from {pretrain_mm_mlp_adapter}. Incompatible keys: {incompatible_keys}")
            incompatible_keys = self.vision_resampler.load_state_dict(get_w(mm_projector_weights, "vision_resampler"), strict=False)
            rank0_print(f"Loaded vision resampler weights from {pretrain_mm_mlp_adapter}. Incompatible keys: {incompatible_keys}")
            
            
#             self.vision_select.data = mm_projector_weights["model.vision_select"]
            
#             self.text_mlp.load_state_dict(get_w(mm_projector_weights, "text_mlp"))
            
#             self.text_select_model.load_state_dict(get_w(mm_projector_weights, "text_select_model"),strict=False)
            #self.vision_tower.load_state_dict(get_w(mm_projector_weights, "vision_tower"),strict=False)

def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of the image (height, width).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    # Compute aspect ratios
    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    # Determine padding size and direction
    if original_aspect_ratio > current_aspect_ratio:
        # Padding was added to the height
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding : current_height - padding, :]
    else:
        # Padding was added to the width
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding : current_width - padding]

    return unpadded_tensor

def rotary_position_embedding(q):
    seq_len, dim = q.shape

    position = torch.arange(seq_len, dtype=torch.float).unsqueeze(-1).to(q.device)

    div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float) * -(math.log(1000000.0) / dim)).to(q.device)
    
    pos_emb = position * div_term
    pos_emb = torch.stack([torch.sin(pos_emb), torch.cos(pos_emb)], dim=-1).flatten(-2, -1)
    
    cos_emb = pos_emb[..., 1::2].repeat_interleave(2, dim=-1)
    sin_emb = pos_emb[..., ::2].repeat_interleave(2, dim=-1)
    
    q_alternate = torch.stack([-q[..., 1::2], q[..., ::2]], dim=-1).reshape(q.size())
    
    q_rotated = q * cos_emb + q_alternate * sin_emb

    return q_rotated

class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def get_2dPool(self, image_feature):
        height = width = self.get_vision_tower().num_patches_per_side
        num_frames, num_tokens, num_dim = image_feature.shape
        image_feature = image_feature.view(num_frames, height, width, -1)
        image_feature = image_feature.permute(0, 3, 1, 2).contiguous()
        # image_feature = nn.functional.max_pool2d(image_feature, self.config.mm_spatial_pool_stride)
        if self.config.mm_spatial_pool_mode == "average":
            image_feature = nn.functional.avg_pool2d(image_feature, self.config.mm_spatial_pool_stride)
        elif self.config.mm_spatial_pool_mode == "max":
            image_feature = nn.functional.max_pool2d(image_feature, self.config.mm_spatial_pool_stride)
        else:
            raise ValueError(f"Unexpected mm_spatial_pool_mode: {self.config.mm_spatial_pool_mode}")
        image_feature = image_feature.permute(0, 2, 3, 1)
        image_feature = image_feature.view(num_frames, -1, num_dim)
        return image_feature

    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        #image_features = self.get_model().vision_resampler(image_features, images=images)
        image_features = self.get_model().mm_projector(image_features)
        image_features = self.get_model().vision_resampler(image_features, images=images)
        return image_features

    # def encode_multimodals(self, videos_or_images, video_idx_in_batch, split_sizes=None):
    #     print("####video",videos_or_images.shape)
    #     videos_or_images_features = self.get_model().get_vision_tower()(videos_or_images)
    #     per_videos_or_images_features = torch.split(videos_or_images_features, split_sizes, dim=0)  # tuple, (dim_1, 576, 4096)
    #     all_videos_or_images_features = []

    #     for idx, feat in enumerate(per_videos_or_images_features):

    #         feat = self.get_model().mm_projector(feat)
    #         # Post pooling
    #         if idx in video_idx_in_batch:
    #             feat = self.get_2dPool(feat)
    #         all_videos_or_images_features.append(feat)
    #     return all_videos_or_images_features
    def add_image(self, image_features):
        return torch.repeat_interleave(image_features, repeats=4, dim=0)
    
    def add_video(self, video_features):
        if video_features.size(0)<4:
            last_feature = video_features[-1:]

            repeated_features = last_feature.repeat(4 - video_features.size(0), 1,1,1)
            expanded_x = torch.cat([video_features, repeated_features], dim=0)
            return expanded_x
#         x_flat = video_features.view(video_features.shape[0], -1)  # 形状: [59, 1152*24*24]

#         x1 = x_flat[:-1] 
#         x2 = x_flat[1:] 

#         similarities = F.cosine_similarity(x1, x2, dim=1).float()
        
#         prev_sim = similarities[:-1]  # 与前帧的相似度，形状: [N-2]
#         next_sim = similarities[1:]   # 与后帧的相似度，形状: [N-2]
#         # 计算前后相似度的和
#         sim_sum = prev_sim + next_sim  # 形状: [N-2]
#         # 取相似度和的后 25% 作为阈值
#         threshold = torch.quantile(sim_sum, 0.1)  # 阈值

#         is_mutation = sim_sum < threshold  # 形状: [N-2]

#         mutation_indices = torch.where(is_mutation)[0] + 1  # 加 1 对齐原始索引
        
        
        repeat_counts = torch.ones(video_features.size(0), dtype=torch.long, device=video_features.device)
#        repeat_counts[mutation_indices] = 4  # 突变帧重复 4 次（原始帧 + 3 次重复）
        
        sum_counts=torch.sum(repeat_counts)
        if sum_counts % 4!=0:
            padding_size = 4 - (sum_counts % 4)
            random_indices = torch.randperm(repeat_counts.size(0))[:padding_size].to(video_features.device)
            repeat_counts[random_indices] += 1 
            
        expanded_x = torch.repeat_interleave(video_features, repeat_counts, dim=0)
        
        return expanded_x

    def encode_multimodals(self, videos_or_images, video_idx_in_batch, split_sizes=None):
        #################################################################################
        # if videos_or_images.shape[0] > 360:
        #     random_indices = np.random.choice(videos_or_images.shape[0], size=360, replace=False)
        #     videos_or_images = videos_or_images[random_indices]
        #     split_sizes=videos_or_images.shape[0]
            
        #################################################################################
        # Define the maximum batch size (1024 frames)
        max_batch_size = 300
        num_frames = videos_or_images.shape[0]
        # Initialize a list to store the features from each batch
        videos_or_images_features = []

        # Split videos_or_images into smaller batches if num_frames > max_batch_size
        if num_frames > max_batch_size:
            #print('&&')
            # Calculate the number of batches needed
            num_batches = (num_frames + max_batch_size - 1) // max_batch_size
            for i in range(num_batches):
                start_idx = i * max_batch_size
                end_idx = min((i + 1) * max_batch_size, num_frames)

                # Process each batch separately
                batch_videos_or_images = videos_or_images[start_idx:end_idx]
                batch_features = self.get_model().get_vision_tower()(batch_videos_or_images)
                videos_or_images_features.append(batch_features)

            # Concatenate the features of all batches
            videos_or_images_features = torch.cat(videos_or_images_features, dim=0)
        else:
            videos_or_images_features = self.get_model().get_vision_tower()(videos_or_images)

        per_videos_or_images_features = torch.split(videos_or_images_features, split_sizes, dim=0)  # tuple, (dim_1, 576, 4096)
        all_videos_or_images_features = []
        
        
        for idx, feat in enumerate(per_videos_or_images_features):
            #print(feat.shape,end='1\n')
            feat=self.interpolate(feat)
            
            ###########################################################
            if idx in video_idx_in_batch:
                feat=self.add_video(feat)
            else:
                feat=self.add_image(feat)
            bc,ch,h,w=feat.shape
            
            feat = feat.view(bc//4,ch,4,h,w)
            if bc//4>48:
                chunk_size = 48
                chunks = torch.split(feat, chunk_size, dim=0)
                interpolated_chunks = []
                for chunk in chunks:
                    interpolated_chunk=self.get_model().sae(chunk).squeeze(2)
                    interpolated_chunks.append(interpolated_chunk)
                feat = torch.cat(interpolated_chunks, dim=0)
                del interpolated_chunks
                del chunks
            else:
                feat=self.get_model().sae(feat).squeeze(2)
            feat = feat.permute(0, 2, 3, 1).contiguous().flatten(1, 2)
            ###########################################################
            # if idx in video_idx_in_batch:
            #     feat=self.add_video(feat)
            #     bc,ch,h,w=feat.shape
            #     feat = feat.view(bc//4,ch,4,h,w)
            #     if bc//4>48:
            #         chunk_size = 48
            #         chunks = torch.split(feat, chunk_size, dim=0)
            #         interpolated_chunks = []
            #         for chunk in chunks:
            #             interpolated_chunk=self.get_model().sae(chunk).squeeze(2)
            #             interpolated_chunks.append(interpolated_chunk)
            #         feat = torch.cat(interpolated_chunks, dim=0)
            #         del interpolated_chunks
            #         del chunks
            #     else:
            #         feat=self.get_model().sae(feat).squeeze(2)
            #     feat = feat.permute(0, 2, 3, 1).contiguous().flatten(1, 2)
            ###########################################################
            feat = self.get_model().mm_projector(feat)
            #print(feat.shape,end='4\n')
            # Post pooling
            if idx in video_idx_in_batch:
                #print('************************',idx,video_idx_in_batch)
                feat = self.get_2dPool(feat)
            all_videos_or_images_features.append(feat)
            
        del per_videos_or_images_features
        return all_videos_or_images_features
    ########################################################
    def interpolate(self,image_features):
        b, num_tokens, dim = image_features.shape
        
        #print(str(image_features.shape)+' i\n')
        
        target_h = target_w = int(576**0.5)
        h = w = int(num_tokens**0.5)

        image_features = image_features.view(b, h, w, dim)
        image_features = image_features.permute(0, 3, 1, 2).contiguous()

        # image_features = F.interpolate(
        #     image_features.to(torch.float32),
        #     size=(target_h, target_w),
        #     mode="bilinear",
        #     align_corners=False,
        # ).to(image_features.dtype)
        chunk_size = 36
        chunks = torch.split(image_features, chunk_size, dim=0)
        interpolated_chunks = []
        for chunk in chunks:
            interpolated_chunk = F.interpolate(
                chunk.to(torch.float32),
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            ).to(chunk.dtype)
            interpolated_chunks.append(interpolated_chunk)
        image_features = torch.cat(interpolated_chunks, dim=0)
        del interpolated_chunks
        del chunks

        return image_features
########################################################
    def prepare_inputs_labels_for_multimodal(self, input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities=["image"], image_sizes=None):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]

            video_idx_in_batch = []
            for _ in range(len(modalities)):
                if modalities[_] == "video":
                    video_idx_in_batch.append(_)

            images_list = []
            for image in images:
                if image.ndim == 4:
                    images_list.append(image)
                else:
                    images_list.append(image.unsqueeze(0))

            concat_images = torch.cat([image for image in images_list], dim=0)
            split_sizes = [image.shape[0] for image in images_list]

            image_features = self.encode_multimodals(concat_images, video_idx_in_batch, split_sizes)
            # image_features = torch.split(image_features, split_sizes, dim=0)
            mm_patch_merge_type = getattr(self.config, "mm_patch_merge_type", "flat")
            image_aspect_ratio = getattr(self.config, "image_aspect_ratio", "square")

            if mm_patch_merge_type == "flat":
                image_features = [x.flatten(0, 1) for x in image_features]
            
            elif mm_patch_merge_type== "unires":
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    # rank0_print(f"Initial feature size : {image_feature.shape}")
                    if image_idx in video_idx_in_batch:  # video operations
                        image_feature = image_feature.flatten(0, 1)
                    elif image_feature.shape[0] > 1:
                        # base image feature is never used in unires
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]
                        # rank0_print(f"Before pool : {image_feature.shape}")
                        height = width = self.get_vision_tower().num_patches_per_side
                        assert height * width == base_image_feature.shape[0]
                        if hasattr(self.get_vision_tower(), "image_size"):
                            vision_tower_image_size = self.get_vision_tower().image_size
                        else:
                            raise ValueError("vision_tower_image_size is not found in the vision tower.")
                        #num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, vision_tower_image_size)
                        #image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                        # Assume 2*2 patches
                        # After this, [2,2, 24,24, 4096]
                        kernel_size = mm_patch_merge_type.split("avgpool")[-1].split("x")[-1]
                        kernel_size = 2
                        image_feature = image_feature.view(image_feature.shape[0], height, width, -1) # [4, 24, 24, 4096]
                        image_feature = image_feature.permute(0, 3, 1, 2).contiguous() # [4, 4096, 24, 24]
                        image_feature = nn.functional.avg_pool2d(image_feature, kernel_size) # [4, 4096, 12, 12]
                        image_feature = image_feature.flatten(2, 3) # [4, 4096, 144]
                        image_feature = image_feature.permute(0, 2, 1).contiguous() # [4, 144, 4096]
                        image_feature = image_feature.flatten(0, 1) # [576, 4096]
                        # rank0_print(f"After pool : {image_feature.shape}")
                    else:
                        # for text only data, there is a placeholder image feature that is actually never used. 
                        image_feature = image_feature[0]
                        # rank0_print(f"After here : {image_feature.shape}")
                    new_image_features.append(image_feature)

                image_features = new_image_features
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        else:
            error_message = """
            Something is wrong with the input shape. Most likely, you did not wrap the video input in a list:
            This is correct:
                model.generate(input_ids, images=[video_tensor],  modalities=["video"], **gen_kwargs)
            This is wrong:
                model.generate(input_ids, images=video_tensor,  modalities=["video"], **gen_kwargs)
            """
            raise ValueError(error_message)
            # image_features = self.encode_images(images)

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(self.config, "mm_use_im_start_end", False):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1 : image_token_indices[i + 1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i] + 1 : image_token_indices[i + 1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            # import pdb; pdb.set_trace()
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, "tokenizer_model_max_length", None)

        new_input_embeds = [x[:tokenizer_model_max_length] for x, modality in zip(new_input_embeds, modalities)]
        new_labels = [x[:tokenizer_model_max_length] for x, modality in zip(new_labels, modalities)]
        # TODO: Hard code for control loss spike
        # if tokenizer_model_max_length is not None:
        #     new_input_embeds = [x[:4096] if modality != "video" else x[:tokenizer_model_max_length] for x, modality in zip(new_input_embeds, modalities)]
        #     new_labels = [x[:4096] if modality != "video" else x[:tokenizer_model_max_length] for x, modality in zip(new_labels, modalities)]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, "tokenizer_padding_side", "right") == "left":
                new_input_embeds_padded.append(torch.cat((torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device), cur_new_embed), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((cur_new_embed, torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None
        if getattr(self.config, "use_pos_skipping", False) and self.training:
            position_ids = torch.arange(new_input_embeds.size(1), device=new_input_embeds.device).unsqueeze(0).to(new_input_embeds.device)
            split_position = random.randint(0, new_input_embeds.size(1))
            left_add = random.randint(0, self.config.pos_skipping_range)
            right_add = random.randint(left_add, self.config.pos_skipping_range)
            position_ids[:, :split_position] += left_add
            position_ids[:, split_position:] += right_add
        # import pdb; pdb.set_trace()
        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location="cpu")
                embed_tokens_weight = mm_projector_weights["model.embed_tokens.weight"]
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")

        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
    

    def get_image_features(self, input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities=["image"], image_sizes=None):
        vision_tower = self.get_vision_tower()
        
        if images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels
        
        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]

            video_idx_in_batch = []
            for _ in range(len(modalities)):
                if modalities[_] == "video":
                    video_idx_in_batch.append(_)

            images_list = []
            for image in images:
                if image.ndim == 4:
                    images_list.append(image)
                else:
                    images_list.append(image.unsqueeze(0))
            #print(len(images_list),images_list[0].shape)

            concat_images = torch.cat([image for image in images_list], dim=0)
            split_sizes = [image.shape[0] for image in images_list]
            #print("##########",concat_images.shape) # 16,3,336,336
            
            #print(video_idx_in_batch)
            image_features = self.encode_multimodals(concat_images, video_idx_in_batch, split_sizes)    #16,144,3584
   
            #print("$$$$$$$$$$$",len(image_features),image_features[0].shape)
            mm_patch_merge_type = getattr(self.config, "mm_patch_merge_type", "flat")
            image_aspect_ratio = getattr(self.config, "image_aspect_ratio", "square")
            # print(mm_patch_merge_type)  #unires
            #print(str(len(image_features))+str(image_features[0].shape)+' image_fearures--\n')
            visual_drop_score=[]
            new_image_features=[]
            
            if mm_patch_merge_type == "flat":
                
                if image_features[0].ndim>2:
                    image_features = [x.flatten(0, 1) for x in image_features]
            elif mm_patch_merge_type== "unires":
                #print('unires')
                for image_idx, image_feature in enumerate(image_features):
                    # rank0_print(f"Initial feature size : {image_feature.shape}")
                    if image_idx in video_idx_in_batch:  # video operations
                        ##############################################################################
#                         t_dim,token_dim,hidden_dim=image_feature.shape
#                         #print(t_dim,token_dim,hidden_dim)

#                         zero_frame = torch.zeros(1, 144, self.get_model().hidden_size).to(image_feature.device)

#                         before_frame = torch.cat((zero_frame, image_feature[:-1]), dim=0)

#                         before_query=self.get_model().vision_select[:2] 
#                         global_query=self.get_model().vision_select[2:] 

#                         before_frame_query=(before_frame.unsqueeze(0)*before_query[:, None, :].unsqueeze(1)).to(self.dtype)  #50 144 1536 * 4 1 1536->50 
                        
#                         images_expand = image_feature.unsqueeze(0).expand(2,-1, -1, -1)  # [4, 144, 1536] #torch.Size([4, 50, 144, 1536])

#                         before_mat = torch.einsum('bqth,bqth->bqt', images_expand, before_frame_query)
#                         before_mat = before_mat.sum(dim=0)


#                         global_feature=image_feature.mean(1) #1536
#                         global_feature=global_feature.unsqueeze(1)*global_query #4 1536

#                         global_mat=torch.matmul(image_feature,global_feature.transpose(-2, -1)).mean(dim=-1)

#                         sum_query_score=before_mat+global_mat

#                         sum_query_score=sum_query_score.flatten(0)

#                         min_val = torch.min(sum_query_score)
#                         max_val = torch.max(sum_query_score)
#                         sum_query_score = (sum_query_score - min_val) / (max_val - min_val)
                        
                        
#                         #print(sum_query_score)
#                         # with open("visualscore.txt", "w") as file:
#                         #     for jj in sum_query_score:
#                         #         file.write(str(jj.cpu().numpy()))
#                         #         file.write('\n')
                            
                        
#                         #print(t_dim,token_dim,hidden_dim,sum_query_score.shape)
#                         visual_drop_score.append(sum_query_score)

##############################################################################
                        image_feature = image_feature.flatten(0, 1)
                        
                    elif image_feature.shape[0] > 1:
                        # base image feature is never used in unires
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]
                        
                        #print(f"Before pool : {image_feature.shape}")
                        #rank0_print(f"Before pool : {image_feature.shape}")
                        height = width = self.get_vision_tower().num_patches_per_side
                        assert height * width == base_image_feature.shape[0]
                        # if hasattr(self.get_vision_tower(), "image_size"):
                        #     vision_tower_image_size = self.get_vision_tower().image_size
                        # else:
                        #     raise ValueError("vision_tower_image_size is not found in the vision tower.")
                        # num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, vision_tower_image_size)
                        # image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                        
                        #print(f"in pool : {image_feature.shape}") #in pool : torch.Size([6, 8, 24, 24, 1344])

                        kernel_size = mm_patch_merge_type.split("avgpool")[-1].split("x")[-1]
                        kernel_size = 2
                        image_feature = image_feature.view(image_feature.shape[0], height, width, -1) # [4, 24, 24, 4096]
                        image_feature = image_feature.permute(0, 3, 1, 2).contiguous() # [4, 4096, 24, 24]
                        image_feature = nn.functional.avg_pool2d(image_feature, kernel_size) # [4, 4096, 12, 12]
                        image_feature = image_feature.flatten(2, 3) # [4, 4096, 144]
                        image_feature = image_feature.permute(0, 2, 1).contiguous() # [4, 144, 4096]
                        
                        ##############################################################################
#                         t_dim,token_dim,hidden_dim=image_feature.shape
#                         #print(t_dim,token_dim,hidden_dim)

#                         zero_frame = torch.zeros(1, 144, self.get_model().hidden_size).to(image_feature.device)

#                         before_frame = torch.cat((zero_frame, image_feature[:-1]), dim=0)

#                         before_query=self.get_model().vision_select[:2] 
#                         global_query=self.get_model().vision_select[2:] 

#                         before_frame_query=(before_frame.unsqueeze(0)*before_query[:, None, :].unsqueeze(1)).to(self.dtype)  #50 144 1536 * 4 1 1536->50 
                        
#                         images_expand = image_feature.unsqueeze(0).expand(2,-1, -1, -1)  # [4, 144, 1536] #torch.Size([4, 50, 144, 1536])

#                         before_mat = torch.einsum('bqth,bqth->bqt', images_expand, before_frame_query)
#                         before_mat = before_mat.sum(dim=0)


#                         global_feature=image_feature.mean(1) #1536
#                         global_feature=global_feature.unsqueeze(1)*global_query #4 1536

#                         global_mat=torch.matmul(image_feature,global_feature.transpose(-2, -1)).mean(dim=-1)

#                         sum_query_score=before_mat+global_mat

#                         sum_query_score=sum_query_score.flatten(0)

#                         min_val = torch.min(sum_query_score)
#                         max_val = torch.max(sum_query_score)
#                         sum_query_score = (sum_query_score - min_val) / (max_val - min_val)
#                         #print(t_dim,token_dim,hidden_dim,sum_query_score.shape)
#                         visual_drop_score.append(sum_query_score)

            ##############################################################################
                        #new_image_features.append(images.flatten(0, 1)) 

                        #new_image_features.append(images.flatten(0, 1)[top_indices])
                        image_feature = image_feature.flatten(0, 1)
                        
                    else:
                        ##############################################################################
#                         t_dim,token_dim,hidden_dim=image_feature.shape
#                         #print(t_dim,token_dim,hidden_dim)

#                         zero_frame = torch.zeros(1, 144, self.get_model().hidden_size).to(image_feature.device)

#                         before_frame = torch.cat((zero_frame, image_feature[:-1]), dim=0)

#                         before_query=self.get_model().vision_select[:2] 
#                         global_query=self.get_model().vision_select[2:] 

#                         before_frame_query=(before_frame.unsqueeze(0)*before_query[:, None, :].unsqueeze(1)).to(self.dtype)  #50 144 1536 * 4 1 1536->50 
                        
#                         images_expand = image_feature.unsqueeze(0).expand(2,-1, -1, -1)  # [4, 144, 1536] #torch.Size([4, 50, 144, 1536])

#                         before_mat = torch.einsum('bqth,bqth->bqt', images_expand, before_frame_query)
#                         before_mat = before_mat.sum(dim=0)


#                         global_feature=image_feature.mean(1) #1536
#                         global_feature=global_feature.unsqueeze(1)*global_query #4 1536

#                         global_mat=torch.matmul(image_feature,global_feature.transpose(-2, -1)).mean(dim=-1)

#                         sum_query_score=before_mat+global_mat

#                         sum_query_score=sum_query_score.flatten(0)

#                         min_val = torch.min(sum_query_score)
#                         max_val = torch.max(sum_query_score)
#                         sum_query_score = (sum_query_score - min_val) / (max_val - min_val)
#                         #print(t_dim,token_dim,hidden_dim,sum_query_score.shape)
#                         visual_drop_score.append(sum_query_score)
                        
                        image_feature = image_feature[0]
                        
                    new_image_features.append(image_feature)
                
                image_features = new_image_features
                
            elif mm_patch_merge_type.startswith("spatial"):
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    # FIXME: now assume the image is square, and split to 2x2 patches
                    # num_patches = h * w, where h = w = sqrt(num_patches)
                    # currently image_feature is a tensor of shape (4, num_patches, hidden_size)
                    # we want to first unflatten it to (2, 2, h, w, hidden_size)
                    if image_idx in video_idx_in_batch:  # video operations
                        if "unpad" in mm_patch_merge_type:
                            # image_feature = image_feature.permute(2, 0, 1).contiguous()
                            # image_feature =  torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
                            # image_feature = image_feature.permute(1, 2, 0).contiguous()
                            image_feature = image_feature.flatten(0, 1)
                            image_feature = torch.cat((image_feature, self.model.image_newline[None].to(image_feature.device)), dim=0)

                    elif image_feature.shape[0] > 1:  # multi patches and multi images operations
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]
                        height = width = self.get_vision_tower().num_patches_per_side
                        assert height * width == base_image_feature.shape[0]

                        if "anyres_max" in image_aspect_ratio:
                            matched_anyres_max_num_patches = re.match(r"anyres_max_(\d+)", image_aspect_ratio)
                            if matched_anyres_max_num_patches:
                                max_num_patches = int(matched_anyres_max_num_patches.group(1))

                        if image_aspect_ratio == "anyres" or "anyres_max" in image_aspect_ratio:
                            if hasattr(self.get_vision_tower(), "image_size"):
                                vision_tower_image_size = self.get_vision_tower().image_size
                            else:
                                raise ValueError("vision_tower_image_size is not found in the vision tower.")
                            num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, vision_tower_image_size)
                            image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                        else:
                            image_feature = image_feature.view(2, 2, height, width, -1)

                        if "maxpool2x2" in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = nn.functional.max_pool2d(image_feature, 2)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        elif "unpad" in mm_patch_merge_type and "anyres_max" in image_aspect_ratio and matched_anyres_max_num_patches:
                            unit = image_feature.shape[2]
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            c, h, w = image_feature.shape
                            times = math.sqrt(h * w / (max_num_patches * unit**2))
                            if times > 1.1:
                                image_feature = image_feature[None]
                                image_feature = nn.functional.interpolate(image_feature, [int(h // times), int(w // times)], mode="bilinear")[0]
                            image_feature = torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        elif "unpad" in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            image_feature = torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        else:
                            image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                            image_feature = image_feature.flatten(0, 3)
                        if "nobase" in mm_patch_merge_type:
                            pass
                        else:
                            image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                    else:  # single image operations
                        image_feature = image_feature[0]
                        if "unpad" in mm_patch_merge_type:
                            image_feature = torch.cat((image_feature, self.model.image_newline[None]), dim=0)

                    new_image_features.append(image_feature)
                image_features = new_image_features
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        else:
            error_message = """
            Something is wrong with the input shape. Most likely, you did not wrap the image or video input in a list:
            This is correct:
                model.generate(input_ids, images=[video_tensor],  modalities=["video"], **gen_kwargs)
                model.generate(input_ids, images=[image_tensor],  modalities=["image"], **gen_kwargs)
            This is wrong:
                model.generate(input_ids, images=video_tensor,  modalities=["video"], **gen_kwargs)
                model.generate(input_ids, images=image_tensor,  modalities=["image"], **gen_kwargs)
            """
            raise ValueError(error_message)
            
        #print(image_features[0].shape,end='2\n')
        
        token_score_features=[]
        for text_ids in range(len(input_ids)):
            #print(len(input_ids),end='text\n')
            #######################################################################
  
            text_per=input_ids[text_ids]

            num_images = (text_per == IMAGE_TOKEN_INDEX).sum()

            image_token_indices = (
                [-1]
                + torch.where(text_per == IMAGE_TOKEN_INDEX)[0].tolist()
                + [text_per.shape[0]]
            )

            cur_input_ids_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(
                    text_per[
                        image_token_indices[i] + 1 : image_token_indices[i + 1]
                    ]
                )
                
            del text_per
            cur_input_ids_noim=torch.cat(cur_input_ids_noim)
            outputs_text_select  = self.get_model().embed_tokens(cur_input_ids_noim)
            
            
            
            
            outputs_text_select=self.get_model().text_mlp(outputs_text_select)

            t_sum,chan_sum=image_features[text_ids].shape
            image_feature_per=image_features[text_ids]

            select_mat=torch.matmul(image_feature_per,outputs_text_select.transpose(0, 1)).mean(dim=-1)
            #######################################################################
            min_val = torch.min(select_mat)
            max_val = torch.max(select_mat)
            select_mat = (select_mat - min_val) / (max_val - min_val)

            #######################################################################
            #select_mat=self.get_model().text_gamma*select_mat+visual_drop_score[text_ids]*(1-self.get_model().text_gamma)
            #######################################################################
            #print(select_mat)
            
            # with open("textscore.txt", "w") as file:
            #     for jj in select_mat:
            #         file.write(str(jj.cpu().numpy()))
            #         file.write('\n')
                                
                                
            token_score_features.append(select_mat)
            
        for image_ind in range(len(image_features)):
            typ=image_features[image_ind].dtype
            image_features[image_ind]=rotary_position_embedding(image_features[image_ind]).to(typ)

        new_input_embeds = []

        for image_idx in range(len(image_features)):
            #print(len(image_features),end='img\n')
            image_per=image_features[image_idx]+token_score_features[image_idx].unsqueeze(1)
            
            t_sum,chan_sum=image_per.shape
            
            save_time_sum=int(t_sum*1)
            if t_sum>9600:
                save_time_sum=9600
            else:
                save_time_sum=int(t_sum*1)
            
            save_time_sum=save_time_sum-save_time_sum % 16
            
            
            _, top_indices = torch.topk(token_score_features[image_idx],save_time_sum)
            top_indices=torch.tensor(sorted(top_indices))
            
            image_per=image_per[top_indices]
            
            new_input_embeds.append(image_per)
        
        image_features=new_input_embeds
        
        return image_features





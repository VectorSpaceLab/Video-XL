import torch
import torch.nn.functional as F
from torch import nn
from typing import Optional, Tuple, Union, Dict
from PIL import Image
from functools import partial, reduce
from transformers import SiglipImageProcessor, SiglipVisionConfig, SiglipVisionModel

from .base_encoder import BaseVisionTower
import torch.distributed as dist
# --data_path /share/shuyan/video_traindata/anno/\{cinepine_order\}.json \
#     --image_folder /share/shuyan/video_traindata/Bunny-v1_0-data/finetune/images \
#     --video_folder /share/shuyan/video_traindata \
def rank0_print(*args):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(f"Rank {dist.get_rank()}: ", *args)
    else:
        print(*args)

        
from transformers.image_processing_utils import BatchFeature, get_size_dict
from transformers.image_transforms import (
    convert_to_rgb,
    normalize,
    rescale,
    resize,
    to_channel_dimension_format,
)
from transformers.image_utils import (
    ChannelDimension,
    PILImageResampling,
    to_numpy_array,
)
class SigLipImageProcessor:
    def __init__(self, image_mean=(0.5, 0.5, 0.5), image_std=(0.5, 0.5, 0.5), size=(384, 384), crop_size: Dict[str, int] = None, resample=PILImageResampling.BICUBIC, rescale_factor=1 / 255, data_format=ChannelDimension.FIRST):
        crop_size = crop_size if crop_size is not None else {"height": 384, "width": 384}
        crop_size = get_size_dict(crop_size, default_to_square=True, param_name="crop_size")

        self.image_mean = image_mean
        self.image_std = image_std
        self.size = size
        self.resample = resample
        self.rescale_factor = rescale_factor
        self.data_format = data_format
        self.crop_size = crop_size

    def preprocess(self, images, return_tensors):
        if isinstance(images, Image.Image):
            images = [images]
        else:
            # to adapt video data
            images = [to_numpy_array(image) for image in images]
            assert isinstance(images, list)

        transforms = [
            convert_to_rgb,
            to_numpy_array,
            partial(resize, size=self.size, resample=self.resample, data_format=self.data_format),
            partial(rescale, scale=self.rescale_factor, data_format=self.data_format),
            partial(normalize, mean=self.image_mean, std=self.image_std, data_format=self.data_format),
            partial(to_channel_dimension_format, channel_dim=self.data_format, input_channel_dim=self.data_format),
        ]

        images = reduce(lambda x, f: [*map(f, x)], transforms, images)
        
        data = {"pixel_values": images}

        return BatchFeature(data=data, tensor_type=return_tensors)
    
class SigLipVisionTower(BaseVisionTower):
    def __init__(self, vision_tower_name, vision_tower_cfg, delay_load=False):
        super(SigLipVisionTower, self).__init__(vision_tower_name, vision_tower_cfg, delay_load)
        
        res, interp = 384, 576
        self.vision_tower_name = vision_tower_name
        self._image_size = res if res is not None else 512
        self.unfreeze_mm_vision_tower = getattr(vision_tower_cfg, "unfreeze_mm_vision_tower", False)
            
        if not delay_load:
            rank0_print(f"Loading vision tower: {vision_tower_name}")
            self.load_model()
        elif getattr(vision_tower_cfg, "unfreeze_mm_vision_tower", False):
            # TODO: better detector is needed.
            rank0_print(f"The checkpoint seems to contain `vision_tower` weights: `unfreeze_mm_vision_tower`: True.")
            self.load_model()
        elif hasattr(vision_tower_cfg, "mm_tunable_parts") and "mm_vision_tower" in vision_tower_cfg.mm_tunable_parts:
            rank0_print(f"The checkpoint seems to contain `vision_tower` weights: `mm_tunable_parts` contains `mm_vision_tower`.")
            self.load_model()
        else:
            self.cfg_only = self.config

    def load_model(self, device_map=None):
        self.vision_model = "siglip"
        # clip_model, processor = create_model_from_pretrained(self.vision_tower_name)
        self.vision_tower = SiglipVisionModel.from_pretrained(self.vision_tower_name)

        # self.vision_tower = clip_model.visual.trunk
        self.vision_tower.output_tokens = True
        
        self._hidden_size = self.vision_tower.config.hidden_size

        self.image_processor = SigLipImageProcessor()
        
        del self.vision_tower.vision_model.encoder.layers[-1:]
        self.vision_tower.vision_model.head = nn.Identity()

        self.vision_tower.requires_grad_(self.unfreeze_mm_vision_tower)
        self.is_loaded = True

    def _forward(self, images):
        with torch.set_grad_enabled(self.unfreeze_mm_vision_tower):
            image_features = self.vision_tower.forward(
                images.to(device=self.device, dtype=self.dtype),
                output_hidden_states=True,
            ).hidden_states[-1]
            return image_features
    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        for p in self.vision_tower.parameters():
            return p.dtype

    @property
    def device(self):
        for p in self.vision_tower.parameters():
            return p.device

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (336 // 14) ** 2

    @property
    def num_patches_per_side(self):
        #return self.config.image_size // self.config.patch_size
        return 336//14
        #return 27
        # return self.model_config["vision_cfg"]["image_size"] // self.model_config["vision_cfg"]["patch_size"]

    @property
    def image_size(self):
        return 384
        #return self.config.image_size
from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseVisionTower(nn.Module):
    def __init__(self, vision_tower_name, vision_tower_cfg, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower_name
        self.delay_load = delay_load

    @abstractmethod
    def load_model(self, device_map=None):
        raise NotImplementedError("Subclasses must implement load_model")

    @abstractmethod
    def _forward(self, images):
        raise NotImplementedError("Subclasses must implement forward")

    def forward(self, images):
        if type(images) is list:
            image_features = [self._forward(image.unsqueeze(0)) for image in images]
        else:
            image_features = self._forward(images)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        # Dynamically infer the dtype from the first parameter, if not explicitly specified
        if hasattr(self.vision_tower, "dtype"):
            return self.vision_tower.dtype
        else:
            params = list(self.vision_tower.parameters())
            return (
                params[0].dtype if len(params) > 0 else torch.float32
            )  # Default to torch.float32 if no parameters

    @property
    def device(self):
        # Dynamically infer the device from the first parameter, if not explicitly specified
        if hasattr(self.vision_tower, "device"):
            return self.vision_tower.device
        else:
            params = list(self.vision_tower.parameters())
            return (
                params[0].device if len(params) > 0 else torch.device("cpu")
            )  # Default to CPU if no parameters
    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only
    @property
    def hidden_size(self):
        try:
            return self.config.hidden_size
        except:
            return self._hidden_size
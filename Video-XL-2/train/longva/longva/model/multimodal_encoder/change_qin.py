from siglip_encoder_navit import SigLipVisionTower, SigLipVisionModel, SigLipImageProcessor
import torch
import os
import pdb
import sys
from PIL import Image
import pdb

sys.path.append('/share/LXRlxr0_0/code/videoxl2/videoxl2/longva/longva/model/multimodal_encoder')

device_map = 'cuda:0'
encoder = SigLipVisionModel.from_pretrained('/share/minghao/Models/siglip-so400m-patch14-384', device_map=device_map, torch_dtype=torch.bfloat16)

images_dir = '/share/minghao/Projects/Kimi-VL/test-frames'
extracted_frame_paths = os.listdir(images_dir)
extracted_frame_paths = [os.path.join(images_dir, tmp) for tmp in extracted_frame_paths]
extracted_frame_paths = extracted_frame_paths[:8]

loaded_images = []
for frame_path in extracted_frame_paths:
    if os.path.exists(frame_path):
        loaded_images.append(Image.open(frame_path))
    else:
        print(f"Warning: Saved frame path not found: {frame_path}")

preprocessor = SigLipImageProcessor()
siglip_inps = preprocessor.preprocess(loaded_images, 'pt')

images_inp = siglip_inps.data['pixel_values']
images_inp = images_inp.to(device_map,dtype=torch.bfloat16)
print(f'images_inp.shape: {images_inp.shape}')

out = encoder(images_inp)
print(out.pooler_output.shape)
print(out.last_hidden_state.shape)
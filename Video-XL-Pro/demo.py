from videoxlpro.videoxlpro.model.builder import load_pretrained_model
from videoxlpro.videoxlpro.mm_utils import tokenizer_image_token, process_images,transform_input_id
from videoxlpro.videoxlpro.constants import IMAGE_TOKEN_INDEX
from PIL import Image
from decord import VideoReader, cpu
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

# fix seed
torch.manual_seed(0)

model_path="/share/LXRlxr0_0/Video-XL-Pro-3B"
video_path="/share/junjie/code/videofactory/Evaluation_LVBench/MLVU_Test/video/test_sports_7.mp4"


max_frames_num = 128
gen_kwargs = {"do_sample": True, "temperature": 0.01, "top_p": 0.001, "num_beams": 1, "use_cache": True, "max_new_tokens": 128}
tokenizer, model, image_processor, _ = load_pretrained_model(model_path, None, "llava_qwen", device_map="cuda:0")

prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<image>\nDescribe this video,<|im_end|>\n<|im_start|>assistant\n"

input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)

vr = VideoReader(video_path, ctx=cpu(0))

total_frame_num = len(vr)

uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)

frame_idx = uniform_sampled_frames.tolist()

frames = vr.get_batch(frame_idx).asnumpy()


video_tensor = image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].to(model.device, dtype=torch.float16)


with torch.inference_mode():
    output_ids = model.generate(input_ids, images=[video_tensor],  modalities=["video"], **gen_kwargs)
    
ind=torch.where(output_ids[0] == 198)[0][-1]
output_ids= output_ids[:,ind+1:]

outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

print(outputs)
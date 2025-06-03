from longva.longva.model.builder import load_pretrained_model
from longva.longva.mm_utils import tokenizer_image_token, process_images,transform_input_id
from longva.longva.constants import IMAGE_TOKEN_INDEX
from PIL import Image
from decord import VideoReader, cpu
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

# fix seed
torch.manual_seed(0)

def load_video(video_path, max_frames_num):
    if type(video_path) == str:
        vr = VideoReader(video_path, ctx=cpu(0))
    else:
        vr = VideoReader(video_path[0], ctx=cpu(0))
    total_frame_num = len(vr)

    fps = vr.get_avg_fps()

    uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
    frame_idx = uniform_sampled_frames.tolist()
    spare_frames = vr.get_batch(frame_idx).asnumpy()

    frame_idx = uniform_sampled_frames.tolist()

    spare_frames = vr.get_batch(frame_idx).asnumpy()

    timestamps = [round(frame_index / fps, 1) for frame_index in frame_idx]
    #print(timestamps)
    return spare_frames,timestamps
# model_path = "/share/junjie/shuyan/new2_beacon/checkpoints/longva_qwen_retrain_all_onemachinenew"
# model_path="/share/junjie/shuyan/new2_beacon/checkpoints/longva_qwen_abnextqa_unfreeze_onestage"
model_path="/share/LXRlxr0_0/code/videoxl2/videoxl2/checkpoints/videoxl2_0401"

video_path="/share/junjie/code/videofactory/Evaluation_LVBench/MLVU_Test/video/test_sports_7.mp4"

#8600
max_frames_num = 12# you can change this to several thousands so long you GPU memory can handle it :)
gen_kwargs = {"do_sample": True, "temperature": 0.01, "top_p": 0.001, "num_beams": 1, "use_cache": True, "max_new_tokens": 128}
tokenizer, model, image_processor, _ = load_pretrained_model(model_path, None, "llava_qwen", device_map="cuda:0")
# model.config.beacon_ratio=[8]
# print(model)
#video input
# prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<image>\nPlease describe the content of this video in order.<|im_end|>\n<|im_start|>assistant\n"
prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<image>\nDescribe this video,<|im_end|>\n<|im_start|>assistant\n"

input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)


frames,times= load_video(video_path,max_frames_num)
time_stamps=[]
token_frames_sum=(len(times)+3)//4
compress_frame = times[::4]
time_embedding = []
for time in compress_frame:
    #time="{:06.1f}".format(time)
    item = f"Time {time}s:"
    time_embedding.append(tokenizer(item).input_ids)
    time_embedding.append([151654]*144)

time_embedding = [item for sublist in time_embedding for item in sublist]
time_embedding = torch.tensor(time_embedding, dtype=torch.long).to(model.device)
time_stamps.append(time_embedding)


video_tensor = image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].to(model.device, dtype=torch.float16)


with torch.inference_mode():
    output_ids = model.generate(input_ids, images=[video_tensor],time_embedding=time_stamps, modalities=["video"], **gen_kwargs)
    
    
#output_ids = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

print(outputs)
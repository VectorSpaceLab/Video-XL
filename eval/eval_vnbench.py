from videoxl.model.builder import load_pretrained_model
from videoxl.mm_utils import tokenizer_image_token, process_images,transform_input_id
from videoxl.constants import IMAGE_TOKEN_INDEX
from PIL import Image
from decord import VideoReader, cpu
import torch
import numpy as np
import json
from tqdm import tqdm
import os
import argparse
from PIL import Image
import random
import numpy as np
from torch.utils.data import Dataset
from moviepy.editor import VideoFileClip, concatenate_videoclips
import cv2
import re
import argparse



class VNBench(Dataset):
    def __init__(self, data_dir):
        self.data_list = []
     
        with open(data_dir, 'r') as f:
            json_data = json.load(f)
        for data in json_data:
            self.data_list.append({
                'data': data
            })
    
    
        
    def __len__(self):
        return len(self.data_list)
    

    def qa_template(self, data):
        question = f"Question: {data['question']}\n"
        question += "Options:\n"
        answer = data['gt']
        answer_idx = -1
        for idx, c in enumerate(data['options']):
            question += f"({chr(ord('A') + idx)}) {c}\n"
            if c == answer:
                answer_idx = idx
        question = question.rstrip()
        answer = f"({chr(ord('A') + answer_idx)}) {answer}"
        return question, answer

    def __getitem__(self, idx):
        video_path = os.path.join("/share/junjie/shuyan/VNBench/video", self.data_list[idx]['data']['video'].split("/")[-1])
        question, answer = self.qa_template(self.data_list[idx]['data'])
            
        return {
            'video': video_path, 
            'question': question, 
            'answer': answer,
            'type':self.data_list[idx]['data']['type'],
            'try':self.data_list[idx]['data']['try']
        }



# fix seed
torch.manual_seed(0)

model_path = "VideoXL_weight_8"


 # you can change this to several thousands so long you GPU memory can handle it :)
gen_kwargs = {"do_sample": False, "temperature": 0, "top_p": None, "num_beams": 1, "use_cache": True, "max_new_tokens": 1024}
tokenizer, model, image_processor, _ = load_pretrained_model(model_path, None, "llava_qwen", device_map="cuda:0")


data_dir = f"/VNBench/VNBench-main-4try.json"
result_path="submit/vnbench.jsonl"
dataset = VNBench(data_dir)

last_name = ""
name_counter = {}  # 用来记录每个 video_name 的计数

result=[]
for example in tqdm(dataset):
    model.memory.reset()
    video_path = example["video"]
    inp = example["question"] 
    task_type = example["type"]
    task_try = example["try"]
    gt = example['answer']
    video_name = video_path.split("/")[-1].replace(".mp4", "")

    # 如果这个 video_name 之前已经出现过，则计数+1
    if video_name in name_counter:
        name_counter[video_name] += 1
    else:
        name_counter[video_name] = 0  # 第一次出现，初始化计数

    # 根据计数创建带编号的 question_id
    question_id = f"{video_name}_{name_counter[video_name]}"
  


    #video input
    prompt1="<|im_start|>system\nCarefully watch this video and pay attention to every detail. Based on your observations, select the best option that accurately addresses the question.<|im_end|>\n<|im_start|>user\n<image>\n"
    prompt2=inp
    prompt3="\n<|im_end|>\n<|im_start|>assistant\n"
    prompt=prompt1 + prompt2 + prompt3
    
    print("#####",prompt)
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)

    # 1fps sampling
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frame_num = len(vr)
    fps = vr.get_avg_fps()  
    duration = total_frame_num / fps  
    frame_idx = [int(i * fps) for i in range(int(duration)) if i * fps < total_frame_num]
    frames = vr.get_batch(frame_idx).asnumpy()

  
    beacon_skip_first = (input_ids == -200).nonzero(as_tuple=True)[1].item()
  
    num_tokens=144*frames.shape[0]
    beacon_skip_last = beacon_skip_first  + num_tokens
    
    video_tensor = image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].to(model.device, dtype=torch.float16)
    with torch.inference_mode():
        output_ids = model.generate(input_ids, images=[video_tensor],  modalities=["video"],beacon_skip_first=beacon_skip_first,beacon_skip_last=beacon_skip_last, **gen_kwargs)

    if -200 in input_ids:
        transform_input_ids=transform_input_id(input_ids,num_tokens,model.config.vocab_size-1)

    output_ids=output_ids[:,transform_input_ids.shape[1]:]
    pred = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()


    matches_pred = re.findall(r'\((.*?)\)', pred)
    matches_gt = re.findall(r'\((.*?)\)', gt)

    try:
        matches_pred=matches_pred[0]
    except:
        matches_pred=pred
    matches_gt=matches_gt[0]
    
    print("##########")
    print("GT",matches_pred)
    print("Pred",matches_gt)
    print("##########")
    sample={}
    sample["question_id"]=question_id
    sample["pred"]=matches_pred
    sample["gt"]=matches_gt
    sample["type"]=task_type
    sample["try"]=task_try
    sample["prompt"]="PROMPT"
    result.append(sample)

with open(result_path, "w") as f:
    for item in result:
        f.write(json.dumps(item) + "\n")



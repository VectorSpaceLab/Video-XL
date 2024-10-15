import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import numpy as np
from longva.constants import IMAGE_TOKEN_INDEX
from longva.longva.conversation import conv_templates
from longva.model.builder import load_pretrained_model
from longva.mm_utils import tokenizer_image_token, process_images,transform_input_id
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math


from longva.model.builder import load_pretrained_model
from longva.mm_utils import tokenizer_image_token, process_images
from longva.constants import IMAGE_TOKEN_INDEX

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        qs = line["text"]

        # qs = "<image>" + '\n' + qs
        # conv = conv_templates[args.conv_mode].copy()
        # conv.append_message(conv.roles[0], qs)
        # conv.append_message(conv.roles[1], None)
        # prompt = conv.get_prompt()


        # prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<image>\nDescribe the image in details.<|im_end|>\n<|im_start|>assistant\n"
        prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<image>\n{qs}<|im_end|>\n<|im_start|>assistant\n"


        if ".mp4" in image_file:
            new_path=os.path.join(self.image_folder,image_file.replace(".mp4",""))
            num_images =len(os.listdir(new_path))
            frames = []
            for n in range(1, num_images + 1):  # 假设 num_images 是图片数量
                image_path = os.path.join(new_path, f"{n}.png")  # 图片名称为1.png, 2.png, ...
                with Image.open(image_path) as frame:
                    frame = np.array(frame)
                    frames.append(frame)
  
            image_tensor = self.image_processor.preprocess(frames, return_tensors="pt")["pixel_values"]
            size=[0]
            flag=["video"]

        else:
            image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
            image_tensor = process_images([image], self.image_processor, self.model_config)
            size=[image.size]
            flag=["image"]

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        return input_ids, image_tensor, size, flag
    def __len__(self):
        return len(self.questions)


# DataLoader
def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return data_loader


def eval_model(args):
    # Model
 
    tokenizer, model, image_processor, _ = load_pretrained_model(args.model_path, None, "llava_qwen", device_map="cuda:0")
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")


    data_loader = create_data_loader(questions, args.image_folder, tokenizer, image_processor, model.config)
    gen_kwargs = {"do_sample": False, "temperature": 0, "top_p": None, "num_beams": 1, "use_cache": True, "max_new_tokens": 128}
    for (input_ids, image_tensor, size,flag), line in tqdm(zip(data_loader, questions), total=len(questions)):
        model.memory.reset()
        idx = line["question_id"]
        cur_prompt = line["text"]

        image_tensor=image_tensor.squeeze(0).to('cuda', dtype=torch.float16)
        input_ids = input_ids.to(device='cuda', non_blocking=True)

        if flag[0][0]=="image":
            num_tokens=(image_tensor.shape[1]-1) *144
            with torch.inference_mode():
                output_ids = model.generate(input_ids, images=image_tensor, image_sizes=size, modalities=["image"],**gen_kwargs)
        elif flag[0][0]=="video":
            num_tokens=(image_tensor.shape[0]) *144
            with torch.inference_mode():
                output_ids = model.generate(input_ids, images=[image_tensor], modalities=["video"],**gen_kwargs)
            
            
        transform_input_ids=transform_input_id(input_ids,num_tokens,model.config.vocab_size-1)

        output_ids=output_ids[:,transform_input_ids.shape[1]:]
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": "long_qwen",
                                   "metadata": {}}) + "\n")
        # ans_file.flush()
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--model-type", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default=None)
    parser.add_argument("--question-file", type=str, default=None)
    parser.add_argument("--answers-file", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()

    eval_model(args)

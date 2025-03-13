from videoxl.model.builder import load_pretrained_model
from videoxl.mm_utils import tokenizer_image_token, process_images, transform_input_id
from videoxl.constants import IMAGE_TOKEN_INDEX
from PIL import Image
import torch
import numpy as np
import time
from base import ViLLMBaseModel
import pdb
torch.manual_seed(32)

class Beacon(ViLLMBaseModel):
    def __init__(self, model_args):
        super().__init__(model_args["model_path"], model_args["device"])
        assert (
            "model_path" in model_args
            and "device" in model_args
        )

        device_id = model_args["device"]
        device_map = f'cuda:{device_id}'
        self.model_name = "Beacon"
        print(f'device_map: {device_map}')
        
        import traceback
        try:
            self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(model_args["model_path"], None, "llava_qwen", device_map=device_map, attn_implementation='sdpa')
        except Exception as e:
            print(f"An error occurred: {e}")
            traceback.print_exc()
            pdb.set_trace()

        overwrite_beacon_ratio = model_args['beacon_ratio']
        self.model.model.config.beacon_ratio=[overwrite_beacon_ratio] # 32 
        print(f'self.model.config.beacon_ratio: {self.model.config.beacon_ratio}')

        self.needle_paths = model_args['needle_paths']
        image_read_s_time = time.time()
        self.needles = []
        for needle_path in self.needle_paths:
            print(needle_path)
            img = Image.open(needle_path).convert("RGB")
            img = np.array(img)
            self.needles.append(img)
        image_read_e_time = time.time()
        print(f'image read time:{image_read_e_time-image_read_s_time}')
    
    def init_select_frames_tensors(self, context_lengths, pre_selecte_frames):
        self.pre_selecte_frames_tensors = {}
        select_frame_s_time = time.time()
        for context_length in context_lengths:
            frames = pre_selecte_frames[context_length]
            video_tensor = self.image_processor.preprocess(frames, return_tensors="pt")["pixel_values"]
            self.pre_selecte_frames_tensors[context_length] = video_tensor

        select_frame_e_time = time.time()
        print(f'pre tensor time:{select_frame_e_time-select_frame_s_time}')

    # TODO 传入 context length 和 depth 用于拼接； needle idx，指定是哪个needle
    def generate(self, instruction, context_length, depth_percent, needle_idx, gt_chunk_idx=None):
        
        gen_kwargs = {"do_sample": False, "use_cache": False, "max_new_tokens": 1024}
        preprompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
        postprompt = "<|im_end|>\n<|im_start|>assistant\nBest option:("
        prompt = preprompt + "<image>" + instruction + postprompt
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.model.device)

        video_tensor = self.pre_selecte_frames_tensors[context_length]
        video_tensor = video_tensor.to(self.model.device, dtype=torch.float16)

        needle_image = self.needles[needle_idx]
        images_tensor_s_time = time.time()
        images_tensor = self.image_processor.preprocess(needle_image, return_tensors="pt")["pixel_values"].to(self.model.device, dtype=torch.float16)
        images_tensor_e_time = time.time()
       
        insert_point = int( len(video_tensor)*(depth_percent*0.01) )
        cat_s_time = time.time()
        video_tensor = torch.cat([video_tensor[:insert_point], images_tensor, video_tensor[insert_point:]], dim=0)
        cat_e_time = time.time()
        
        self.model.memory.reset()
        beacon_skip_first = (input_ids == -200).nonzero(as_tuple=True)[1].item()

        num_tokens=144*len(video_tensor)
        beacon_skip_last = beacon_skip_first  + num_tokens

        with torch.inference_mode():
            output_ids = self.model.generate(input_ids, images=[video_tensor],  modalities=["video"],beacon_skip_first=beacon_skip_first,beacon_skip_last=beacon_skip_last, gt_frame_idx=gt_chunk_idx, **gen_kwargs)

        video_tensor = video_tensor.to('cpu')
        torch.cuda.empty_cache()

        if -200 in input_ids:
            transform_input_ids=transform_input_id(input_ids,num_tokens,self.model.config.vocab_size-1)

        output_ids=output_ids[:,transform_input_ids.shape[1]:]
        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        
        return outputs
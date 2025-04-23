import torch
import numpy as np
from PIL import Image
from decord import VideoReader, cpu
from videoxlpro.videoxlpro.mm_utils import tokenizer_image_token
from videoxlpro.videoxlpro.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

def load_image_processor(model, tokenizer):
    # 配置tokenizer
    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
    if mm_use_im_patch_token:
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    vision_tower.to(device=model.device, dtype=torch.float16)
    image_processor = vision_tower.image_processor
    return image_processor

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

def process_video(video_path,tokenizer,image_processor, model_device, max_frames=128,gen_kwargs=None):
    """
    处理视频并返回处理后的张量
    
    Args:
        video_path: 视频文件路径
        image_processor: 图像处理器
        model_device: 模型所在设备
        max_frames: 最大帧数
        
    Returns:
        处理后的视频张量
    """
    print(f"处理视频: {video_path}")
    
    # 加载视频并采样帧
    frames,times= load_video(video_path,max_frames)
    #print(times)
    time_stamps=[]
    token_frames_sum=(len(times)+3)//4
    compress_frame = times[::4]
    time_embedding = []
    for time in compress_frame:
        time="{:06.1f}".format(time)
        item = f"Time {time}s:"
        time_embedding.append(tokenizer(item).input_ids)
        time_embedding.append([151654]*144)

    time_embedding = [item for sublist in time_embedding for item in sublist]

    time_embedding = torch.tensor(time_embedding, dtype=torch.long).to(model_device)
    time_stamps.append(time_embedding)


    video_tensor = image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].to(model_device, dtype=torch.float16)
    
    
    
    return video_tensor,time_stamps


def generate_response(model, tokenizer, prompt, video_tensor,time_stamps, gen_kwargs=None):
    if gen_kwargs is None:
        gen_kwargs = {
            "do_sample": True,
            "temperature": 0.01,
            "top_p": 0.001,
            "num_beams": 1,
            "use_cache": True,
            "max_new_tokens": 4096
        }
    
    print("生成回答...")
    
    # 处理提示
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)
    
    # 生成回答
    with torch.inference_mode():
        try:
            with torch.inference_mode():
                output_ids = model.generate(input_ids, images=[video_tensor],time_embedding=time_stamps, modalities=["video"], **gen_kwargs)


            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            
            return outputs
        except Exception as e:
            print(f"生成过程中出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return f"错误: {str(e)}" 
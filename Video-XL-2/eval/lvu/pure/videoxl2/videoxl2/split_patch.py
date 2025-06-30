import json
import os
from PIL import Image 
import math
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
import torch
from tqdm import tqdm
import av
import numpy as np
from multiprocessing import Pool, cpu_count


def process_video_with_pyav(video_file):
    container = av.open(video_file)
    stream = container.streams.video[0]
    total_frame_num = stream.frames
    avg_fps = round(stream.average_rate / 1)
    frame_idx = [i for i in range(0, total_frame_num, avg_fps)]
 
    if len(frame_idx) > 128:
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, 128, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()

    video_frames = []
    for index, frame in enumerate(container.decode(video=0)):
        if index in frame_idx:
            video_frames.append(frame.to_rgb().to_ndarray())
            if len(video_frames) == len(frame_idx):  # Stop decoding once we have all needed frames
                break

    video = np.stack(video_frames)
    return video

def process_video(video_path):
    video = process_video_with_pyav(video_path)
    return video


def process_single_video(i, image_folder):
    try:
        video_path = os.path.join(image_folder, i["video"])
        print(f"Processing {video_path}")
        video = process_video(video_path)
        
        new_name = i["video"].split("/")[-1].replace(".mp4", ".npy")
        new_name = new_name.replace("videos", "npy")
        directory_path = os.path.dirname(video_path).replace("videos", "npy")
        
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
        
        new_folder = os.path.join(directory_path, new_name)
        np.save(new_folder, video)
        print(f"Saved {new_folder}")
        
        return video.shape
    except Exception as e:
        print(f"Error processing video {i['video']}: {e}")
        return None


def process_single_video_wrapper(args):
    return process_single_video(*args)


def process_all_videos(data, image_folder, num_workers=None):
    # Set number of workers to available CPU cores if not specified
    if num_workers is None:
        num_workers = cpu_count()

    # Create a list of arguments to pass to the pool
    args = [(i, image_folder) for i in data]
    
    with Pool(num_workers) as pool:
        results = list(tqdm(pool.imap(process_single_video_wrapper, args), total=len(data)))
    
    print(f"Processed {len(results)} videos.")

    
if __name__ == "__main__":
    json_path = "/share/junjie/shuyan/video_traindata/anno/baaicaption3.json"
    image_folder = "/share/junjie/shuyan/video_traindata"
    
    with open(json_path, 'r') as file:
        data = json.load(file)
    
    # Start processing with multiprocessing
    process_all_videos(data, image_folder, num_workers=96)  # You can adjust `num_workers` based on your CPU cores


######################
    # json_path = "/share/junjie/shuyan/video_traindata/anno/baaicaption.json"
    # with open(json_path, 'r') as file:
    #     data = json.load(file)
    
    # result=[]
    # for i in tqdm(data):
    #     folder= os.path.dirname(i["video"]).replace("videos", "npy")
    #     name=i["video"].split("/")[-1].replace(".mp4",".npy")
 
    #     final_path=os.path.join("/share/junjie/shuyan/video_traindata",folder,name)
    #     print(final_path)
    #     if not os.path.exists(final_path):
    #         print("##########")
    #         continue
    #     i["video"]=os.path.join(folder,name)
    #     result.append(i)
    # print(len(result))
    # output_file = "/share/junjie/shuyan/video_traindata/anno/baaicaption_npy.json"
    # with open(output_file, 'w', encoding='utf-8') as f_out:
    #     json.dump(result, f_out, indent=4, ensure_ascii=False)


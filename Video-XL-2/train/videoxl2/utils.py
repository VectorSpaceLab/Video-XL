import datetime
import logging
import logging.handlers
import os
import sys
import numpy as np
import cv2
import requests
import re
import time
from videoxl2.constants import LOGDIR
from func_timeout import func_set_timeout

server_error_msg = "**NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.**"
moderation_msg = "I am sorry. Your input may violate our content moderation guidelines. Please avoid using harmful or offensive content."

handler = None

import torch.distributed as dist

import PIL
from PIL import Image
import re

try:
    import av
    from decord import VideoReader, cpu
except ImportError:
    print("Please install pyav to use video processing functions.")

def read_frames_img(video_path, frames_upbound, fps=2):
    img_files = sorted(
        [f for f in os.listdir(video_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    )
    vlen = len(img_files)
    frame_indices = list(range(0, vlen, fps))

    if len(frame_indices) > frames_upbound:
        frame_indices = np.linspace(0, vlen-1, frames_upbound, dtype=int).tolist()

    imgs = []
    timestamps = []
    for idx in frame_indices:
        img_path = os.path.join(video_path, img_files[idx])
        try:
            with Image.open(img_path) as img:
                img_np = np.array(img.convert("RGB"))  # 使用 PIL 替代 OpenCV
            imgs.append(img_np)
            timestamps.append(round(idx / fps, 1))
        except (IOError, OSError) as e:
            raise RuntimeError(f"Corrupted image {img_path}: {str(e)}")

    if len(imgs) == 0:
        raise RuntimeError(f"No valid frames in {video_path}")

    return np.stack(imgs), vlen // fps, timestamps

def process_video_with_pyav(video_file, data_args):
    # container = av.open(video_file)
    try:
        container = av.open(video_file)
    except av.error as e:
        raise RuntimeError(f"Failed to open video {video_file}: {str(e)}")
    # !!! This is the only difference. Using auto threading
    container.streams.video[0].thread_type = "AUTO"

    video_frames = []
    for packet in container.demux():
        if packet.stream.type == 'video':
            for frame in packet.decode():
                video_frames.append(frame)
    
    if not video_frames:
        print(f"Error: 无法从视频文件 {video_file} 中获取帧")
        return None, None, None
    
    total_frame_num = len(video_frames)
    video_time = video_frames[-1].time  # 视频总时长(秒)
    fps = total_frame_num / video_time if video_time > 0 else 0
    
    video_fps = data_args.video_fps
    avg_fps = round(fps / video_fps)
    frame_idx = [i for i in range(0, total_frame_num, avg_fps)]

    if data_args.frames_upbound > 0 and len(frame_idx) > data_args.frames_upbound:
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, data_args.frames_upbound, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()

    # 获取选中的帧和时间戳
    frames = []
    timestamps = []
    for idx in frame_idx:
        frame = video_frames[idx]
        frames.append(frame.to_ndarray(format="rgb24"))
        timestamps.append(round(frame.time, 1))  # 使用帧的实际时间戳
    
    container.close()

    return video, video_duration_seconds, timestamps

def process_video_with_decord(video_file, data_args):
    if video_file == '/share/shuyan/video_traindata/didemo/57681549@N02_5395130229_88d67d7780.wmv':
        return None, None, []
    # start = time.time()
    # print(f'start process: {video_file}')
    vr = VideoReader(video_file, ctx=cpu(0), num_threads=1)
    total_frame_num = len(vr)
    avg_fps_from_decord = vr.get_avg_fps()
    # 使用用户提供的 video_fps，如果它大于 0，否则使用 decord 提供的平均帧率
    effective_fps = avg_fps_from_decord

    # 如果 effective_fps 仍然是 0，我们无法进行时间戳估算，返回空列表或采取其他策略
    if effective_fps <= 0:
        print("Warning: Effective FPS is 0, cannot estimate timestamps.")
        return None, None, []

    video_fps = data_args.video_fps
    # 根据平均帧率计算帧索引
    step = round(effective_fps / video_fps) if video_fps > 0 and effective_fps > 0 else 1
    frame_idx = [i for i in range(0, total_frame_num, step)]

    fps_upbound = data_args.fps_upbound
    frames_upbound = data_args.frames_upbound

    if fps_upbound is not None:
        higher_fps = min(frames_upbound//len(frame_idx), fps_upbound)
        if higher_fps > video_fps:
            higher_steps = round(effective_fps / higher_fps)
            frame_idx = [i for i in range(0, total_frame_num, higher_steps)]

    if frames_upbound > 0:
        if len(frame_idx) > frames_upbound:
            uniform_sampled_frames = np.linspace(0, total_frame_num - 1, frames_upbound, dtype=int)
            frame_idx = uniform_sampled_frames.tolist()

    # 获取选中的帧和估算的时间戳
    timestamps = [round(idx / effective_fps, 1) for idx in frame_idx]

    video = vr.get_batch(frame_idx).asnumpy()
    vr.seek(0)
    return video, total_frame_num / effective_fps if effective_fps > 0 else 0, timestamps


def rank0_print(*args):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(f"Rank {dist.get_rank()}: ", *args)
    else:
        print(*args)


def build_logger(logger_name, logger_filename):
    global handler

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Set the format of root handlers
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO)
    logging.getLogger().handlers[0].setFormatter(formatter)

    # Redirect stdout and stderr to loggers
    stdout_logger = logging.getLogger("stdout")
    stdout_logger.setLevel(logging.INFO)
    sl = StreamToLogger(stdout_logger, logging.INFO)
    sys.stdout = sl

    stderr_logger = logging.getLogger("stderr")
    stderr_logger.setLevel(logging.ERROR)
    sl = StreamToLogger(stderr_logger, logging.ERROR)
    sys.stderr = sl

    # Get logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Add a file handler for all loggers
    if handler is None:
        os.makedirs(LOGDIR, exist_ok=True)
        filename = os.path.join(LOGDIR, logger_filename)
        handler = logging.handlers.TimedRotatingFileHandler(filename, when="D", utc=True)
        handler.setFormatter(formatter)

        for name, item in logging.root.manager.loggerDict.items():
            if isinstance(item, logging.Logger):
                item.addHandler(handler)

    return logger


class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """

    def __init__(self, logger, log_level=logging.INFO):
        self.terminal = sys.stdout
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ""

    def __getattr__(self, attr):
        return getattr(self.terminal, attr)

    def write(self, buf):
        temp_linebuf = self.linebuf + buf
        self.linebuf = ""
        for line in temp_linebuf.splitlines(True):
            # From the io.TextIOWrapper docs:
            #   On output, if newline is None, any '\n' characters written
            #   are translated to the system default line separator.
            # By default sys.stdout.write() expects '\n' newlines and then
            # translates them so this is still cross platform.
            if line[-1] == "\n":
                self.logger.log(self.log_level, line.rstrip())
            else:
                self.linebuf += line

    def flush(self):
        if self.linebuf != "":
            self.logger.log(self.log_level, self.linebuf.rstrip())
        self.linebuf = ""


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch

    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def violates_moderation(text):
    """
    Check whether the text violates OpenAI moderation API.
    """
    url = "https://api.openai.com/v1/moderations"
    headers = {"Content-Type": "application/json", "Authorization": "Bearer " + os.environ["OPENAI_API_KEY"]}
    text = text.replace("\n", "")
    data = "{" + '"input": ' + f'"{text}"' + "}"
    data = data.encode("utf-8")
    try:
        ret = requests.post(url, headers=headers, data=data, timeout=5)
        flagged = ret.json()["results"][0]["flagged"]
    except requests.exceptions.RequestException as e:
        print(f"######################### Moderation Error: {e} #########################")
        flagged = False
    except KeyError as e:
        print(f"######################### Moderation Error: {e} #########################")
        flagged = False

    return flagged


def pretty_print_semaphore(semaphore):
    if semaphore is None:
        return "None"
    return f"Semaphore(value={semaphore._value}, locked={semaphore.locked()})"

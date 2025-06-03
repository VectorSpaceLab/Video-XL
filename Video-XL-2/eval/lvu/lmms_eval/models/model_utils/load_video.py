import av
import numpy as np
from av.codec.context import CodecContext


# This one is faster
def record_video_length_stream(container, indices):
    frames = []
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return frames


# This one works for all types of video
def record_video_length_packet(container):
    frames = []
    # https://github.com/PyAV-Org/PyAV/issues/1269
    # https://www.cnblogs.com/beyond-tester/p/17641872.html
    # context = CodecContext.create("libvpx-vp9", "r")
    for packet in container.demux(video=0):
        for frame in packet.decode():
            frames.append(frame)
    return frames


def read_video_pyav(video_path, num_frm=8):
    container = av.open(video_path)
    
    # Initialize variables
    frames = []
    frame_indices = []
    frame_pts = []  # 存储每一帧的 pts（时间基准）
    timestamps = []  # 最终返回的时间戳（秒）
    
    # 1. 提取所有帧的 pts（时间戳基准）
    if "webm" not in video_path and "mkv" not in video_path:
        # 对于 MP4，优先尝试 stream 方式读取
        try:
            container = av.open(video_path)
            video_stream = container.streams.video[0]
            total_frames = video_stream.frames
            sampled_frm = min(total_frames, num_frm)
            indices = np.linspace(0, total_frames - 1, sampled_frm, dtype=int)
            
            # 确保包含最后一帧
            if total_frames - 1 not in indices:
                indices = np.append(indices, total_frames - 1)
            
            # 读取帧并记录 pts
            frames = record_video_length_stream(container, indices)
            frame_pts = [frames[i].pts for i in range(len(frames))]
        except:
            # 如果 stream 方式失败，改用 packet 方式
            container = av.open(video_path)
            frames = record_video_length_packet(container)
            total_frames = len(frames)
            sampled_frm = min(total_frames, num_frm)
            indices = np.linspace(0, total_frames - 1, sampled_frm, dtype=int)
            
            if total_frames - 1 not in indices:
                indices = np.append(indices, total_frames - 1)
            
            frames = [frames[i] for i in indices]
            frame_pts = [frame.pts for frame in frames]
    else:
        # 对于 WebM/MKV，直接使用 packet 方式
        container = av.open(video_path)
        frames = record_video_length_packet(container)
        total_frames = len(frames)
        sampled_frm = min(total_frames, num_frm)
        indices = np.linspace(0, total_frames - 1, sampled_frm, dtype=int)
        
        if total_frames - 1 not in indices:
            indices = np.append(indices, total_frames - 1)
        
        frames = [frames[i] for i in indices]
        frame_pts = [frame.pts for frame in frames]
    
    # 2. 计算时间戳（秒）
    if len(frames) > 0:
        # 获取时间基准（time_base），用于计算秒数
        time_base = frames[0].time_base
        timestamps = [round(float(pts * time_base), 1) for pts in frame_pts]
    
    # 3. 转换为 numpy 数组并返回
    frames_array = np.stack([x.to_ndarray(format="rgb24") for x in frames])
    return frames_array, timestamps
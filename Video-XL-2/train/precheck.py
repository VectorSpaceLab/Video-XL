import os
import sys
import json
import random
import argparse
import numpy as np
from tqdm import tqdm
from collections import Counter
from multiprocessing import Pool
import yaml

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_info_path', type=str, default='/share/project/minghao/Proj/Video-XL/Video-XL-2/train/datas/stage4_datas.yaml')
    parser.add_argument('--image_folder', type=str, default='/share/project/minghao/Datas/ImageDatas')
    parser.add_argument('--video_folder', type=str, default='/share/project/minghao/Datas/VideoDatas')
    parser.add_argument('--num_processes', type=int, default=32)
    args = parser.parse_args()
    return args


def load_json(path):
    return json.load(open(path, 'r'))


def check_format(data):
    assert 'id' in data, f'id not in data: {data}'
    assert 'image' in data or 'video' in data, f'image or video not in data: {data}'
    assert 'conversations' in data, f'conversations not in data: {data}'
    assert len(data['conversations']) >= 2 and len(data['conversations'])%2 == 0, f'conversations length not even: {data}'
    assert data['conversations'][0]['from'] == 'human', f'conversations[0] from not human: {data}'
    assert data['conversations'][1]['from'] == 'gpt', f'conversations[1] from not gpt: {data}'
    assert '<image>' in data['conversations'][0]['value'], f'conversations[0] value not contains <image>: {data}'
    
def check_file(data):
    if 'image' in data:
        image_path = os.path.join(image_folder, data['image'])
        assert os.path.exists(image_path), f'image path not exists: {image_path}'
    if 'video' in data:
        video_path = os.path.join(video_folder, data['video'])
        assert os.path.exists(video_path), f'video path not exists: {video_path}'

def check_data(data):
    check_format(data)
    # check_file(data)
    return True

if __name__ == '__main__':
    
    args = get_args()
    image_folder = args.image_folder
    video_folder = args.video_folder

    dataset_info_path = args.dataset_info_path
    dataset_info = yaml.load(open(dataset_info_path, 'r'), Loader=yaml.Loader)

    # num_processes = os.cpu_count() # 使用所有可用CPU核心
    num_processes = args.num_processes
    pool = Pool(processes=num_processes)
    print('=' * 100)
    for datas_info in dataset_info['datasets']:
        dataset_path = datas_info['json_path']
        datas_name = os.path.basename(dataset_path).split('.json')[0]
        print(f'procerssing {datas_name} ...')
        datas = load_json(dataset_path)
        
        # 使用多进程进行检查
        try:
            list(tqdm(pool.imap(check_data, datas), total=len(datas), desc='checking ...'))
            print(f'{datas_name} pass ✅')
        except Exception as e:
            print(f'{datas_name} failed ❌: {str(e)}')
        print('=' * 100)
    
    pool.close()
    pool.join()
    

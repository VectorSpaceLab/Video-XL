import datetime
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import numpy as np
import yaml
from loguru import logger as eval_logger

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file

import pdb

TASK_TYPES = [
'ret_insert1',
'ret_insert2',
'ret_edit1',
'cnt_insert1',
'cnt_edit1',
'cnt_edit2',
'ord_insert1',
'ord_insert2',
'ord_edit1',
]

hf_home="/share/minghao/Datasets/VideoDatasets"
base_cache_dir = os.path.expanduser(hf_home)

with open(Path(__file__).parent / "vnbench.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)

cache_name = yaml.safe_load("".join(safe_data))["dataset_kwargs"]["cache_dir"]

def vnbench_doc_to_visual(doc):
    cache_dir = os.path.join(base_cache_dir, cache_name)
    video_path = doc["video_name"]
    video_path = video_path.split('/')[-1]
    video_path = os.path.join(cache_dir, video_path)
    if os.path.exists(video_path):
        video_path = video_path
    else:
        sys.exit(f"video path:{video_path} does not exist, please check")
    return [video_path]


def vnbench_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    # option_prompt="Carefully watch this video and pay attention to every detail. Based on your observations, select the best option that accurately addresses the question."
    # TODO check the system prompt
    question = doc["question"] 
    full_prompt = question + "\n" + "Answer with the option's letter from the given choices directly.\n"
    return full_prompt


def extract_characters_regex(s):
    s = s.strip()

    pred = s.split('.')[0]

    if 'A' in pred:
        pred = 'A'
    elif 'B' in pred:
        pred = 'B'
    elif 'C' in pred:
        pred = 'C'
    elif 'D' in pred:
        pred = 'D'

    return pred


def vnbench_process_results(doc, results):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name (in this case videomme score), value: metric value
    """
    pred = results[0]
    # print("****************",pred)
    pred_ans = extract_characters_regex(pred)
    video_pure_name = doc['video_name'].split('/')[-1]

    task_type = doc["task_type"]
    data_dict = {"question_id": doc["question"], "task_type": task_type, "pred_answer": pred_ans, "answer": doc["answer"], 'video_pure_name':video_pure_name}

    return {f"vnbench_perception_score": data_dict}


def vnbench_aggregate_results(results):
    """
    Args:
        results: a list of values returned by process_results
    Returns:
        A score
    """
    category2score = {}
    for task_type in TASK_TYPES:
        category2score[task_type] = {"correct": 0, "answered": 0}

    dict_by_video = {}
    for result in results:
        video_pure_name = result['video_pure_name']
        if video_pure_name not in dict_by_video:
            dict_by_video[video_pure_name] = {
                "correct": 0 + (result["pred_answer"] == result["answer"]),
                "task_type": result["task_type"]
            }
        else:
            dict_by_video[video_pure_name]["correct"] = dict_by_video[video_pure_name]["correct"] + (result["pred_answer"] == result["answer"])


    for video_name, info in dict_by_video.items():
        task_type = info["task_type"]
        category2score[task_type]["answered"] += 1
        if info["correct"] == 4:
            category2score[task_type]["correct"] += 1

    # for result in results:
    #     task_type = result["task_type"]
    #     category2score[task_type]["answered"] += 1
    #     category2score[task_type]["correct"] += result["pred_answer"] == result["answer"]

    for task_cate in TASK_TYPES:
        total_correct = 0
        total_answered = 0
        for k, v in category2score.items():
            if task_cate in k:
                total_correct += v["correct"]
                total_answered += v["answered"]
        eval_logger.info(f"Evaluation on Task Categories: {task_cate}: {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%")

    total_correct = 0
    total_answered = 0
    for k, v in category2score.items():
        total_correct += v["correct"]
        total_answered += v["answered"]
    eval_logger.info(f"Overall Performance: {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%")

    return 100 * total_correct / total_answered if total_answered > 0 else 0

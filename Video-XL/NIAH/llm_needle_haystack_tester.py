import glob
import json
import os
from tqdm import tqdm
from datetime import datetime, timezone
import sys
from PIL import Image
import numpy as np
import time
import pdb
from videoxl_modeling import Beacon
from decord import VideoReader, cpu


class LLMNeedleHaystackTester:
    """
    This class is used to test the LLM Needle Haystack.
    """
    def __init__(self,
                 model_name = None,
                 load_gt_chunk = False,
                 overwrite_beacon_ratio=16,
                 model_to_test = None,
                 needle = None,
                 haystack_path = None,
                 needle_desc = None,
                 needle_modality = None,
                 needle_dir = "haystack",
                 retrieval_question = None,
                 answer=None,
                 results_version = 1,
                 context_lengths_min = 300,
                 context_lengths_max = 3000,
                 context_lengths_num_intervals = 100,
                 context_lengths = None,
                 video_depth_percent_min = 0,
                 video_depth_percent_max = 100,
                 video_depth_percent_intervals = 35,
                 video_depth_percents = None,
                 video_depth_percent_interval_type = "linear",
                 save_results = True,
                 print_ongoing_status = True,
                 model_file_location = "",
                 model_path = "",
                 result_dir = "",
                 **kwargs):
        # if not model_to_test:
        #     raise ValueError("A language model must be provided to test.")
        # if not needle or not needle_modality or not haystack_dir or not retrieval_question:
        #     raise ValueError("Needle, Needle_modality, haystack, and retrieval_question must be provided.")
        self.load_gt_chunk = load_gt_chunk
        self.overwrite_beacon_ratio = overwrite_beacon_ratio
        self.needle = needle
        self.haystack_path = haystack_path
        self.needle_desc = needle_desc
        self.needle_modality = needle_modality
        self.needle_dir = needle_dir
        self.retrieval_question = retrieval_question
        self.results_version = results_version
        self.save_results = save_results
        self.print_ongoing_status = print_ongoing_status
        self.testing_results = []
        self.answer=answer
        self.model_file_location = model_file_location
        self.model_path = model_path
        self.result_dir = result_dir

        if context_lengths is None:
            if context_lengths_min is None or context_lengths_max is None or context_lengths_num_intervals is None:
                raise ValueError("Either context_lengths_min, context_lengths_max, context_lengths_intervals need to be filled out OR the context_lengths_list needs to be supplied.")
            else:
                self.context_lengths = np.round(np.linspace(context_lengths_min, context_lengths_max, num=context_lengths_num_intervals, endpoint=True)).astype(int)
        else:
            self.context_lengths = context_lengths

        self.context_lengths = np.array([559, 1118, 1304, 1490])
        self.context_lengths = np.array([187, 559])
        
        if video_depth_percent_interval_type not in [None, "linear", "sigmoid"]:
            raise ValueError("video_depth_percent_interval_type must be either None, 'linear' or 'sigmoid'. If you'd like your own distribution give a list of ints in via video_depth_percent_intervals")

        if video_depth_percents is None:
            if video_depth_percent_min is None or video_depth_percent_max is None or video_depth_percent_intervals is None:
                raise ValueError("Either video_depth_percent_min, video_depth_percent_max, video_depth_percent_intervals need to be filled out OR the video_depth_percents needs to be supplied.")
            
            if video_depth_percent_interval_type == 'linear':
                self.video_depth_percents = np.round(np.linspace(video_depth_percent_min, video_depth_percent_max, num=video_depth_percent_intervals, endpoint=True)).astype(int)
            elif video_depth_percent_interval_type == 'sigmoid':
                self.video_depth_percents = [self.logistic(x) for x in np.linspace(video_depth_percent_min, video_depth_percent_max, video_depth_percent_intervals)]
            else:
                raise ValueError("video_depth_percent_interval_type must be either 'sigmoid' or 'linear' if video_depth_percents is None.")
        else:
            self.video_depth_percents = video_depth_percents
 

        self.model_to_test = model_to_test
        self.model_name = model_name

        print(f'self.video_depth_percents: {self.video_depth_percents}')
        print(f'self.context_lengths: {self.context_lengths}')

        video_read_s_time = time.time()
        self.vr = VideoReader(self.haystack_path, ctx=cpu(0))
        self.total_frames = len(self.vr)
        video_read_e_time = time.time()
        print(f'video read time:{video_read_e_time-video_read_s_time}')
        self.pre_selecte_frames = {}

        select_frame_s_time = time.time()
        for context_length in self.context_lengths:
            uniform_sampled_frames = np.linspace(0, self.total_frames-1, context_length, dtype=int)
            frame_idx = uniform_sampled_frames.tolist()
            frames = self.vr.get_batch(frame_idx).asnumpy() # TODO fix consuming
            self.pre_selecte_frames[context_length] = frames
        select_frame_e_time = time.time()
        print(f'pre select time:{select_frame_e_time-select_frame_s_time}')

        # self.haystack_imgs_dir = '/share/minghao/Projects/NeedleInAVideoHaystack/needlehaystack/newhaystack'
        # video_name = self.haystack_path.split('/')[-1].split('.')[0]
        # this_haystack_imgs_dir = os.path.join(self.haystack_imgs_dir, video_name)

        # VideoReader(self.haystack_path, cpu=)
        
        # video_read_s_time = time.time()
        # self.pre_selecte_frames = {}
        # for context_length in tqdm(self.context_lengths):
        #     this_context_length_dir = os.path.join(this_haystack_imgs_dir, f'{context_length}')
        #     imgs_name = sorted(os.listdir(this_context_length_dir))
        #     frames = []
        #     for img_name in tqdm(imgs_name):
        #         img_path = os.path.join(this_context_length_dir, img_name)
        #         img = Image.open(img_path)
        #         img = np.array(img)
        #         frames.append(img)
            
        #     self.pre_selecte_frames[context_length] = frames

        # video_read_e_time = time.time()
        # print(f'video read time:{video_read_e_time-video_read_s_time}')
    def logistic(self, x, L=100, x0=50, k=.1):
        if x in [0, 100]:
            return x
        x = -k * (x - x0)
        return np.round(L * self.sigmoid(x), 3)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def filter_task(self, tasks):
        filtered_task = []
        
        model_file_location = self.model_file_location
        results_dir = f'/share/minghao/Projects/NeedleInAVideoHaystack/newresults/{model_file_location}'

        for context_length, depth_percent in tasks:
            context_file_location = f'{model_file_location}_modality_{self.needle_modality}_len_{context_length}_depth_{int(depth_percent)}'
            result_path = f'{results_dir}/{context_file_location}_results.json'

            if not os.path.exists(result_path):
                filtered_task.append((context_length, depth_percent))
                print(result_path)
   
        return filtered_task

    def run_test(self):
        # Run through each iteration of context_lengths and depths
        tasks = []
        for context_length in self.context_lengths:
            for depth_percent in self.video_depth_percents:
                tasks.append((int(context_length), int(depth_percent)))

        for task in tasks:
            print(type(task))
            print(type(task[0]), type(task[1]))

        print(f"Total tasks couns: {len(tasks)}")
        tasks = self.filter_task(tasks)
        print(f"Filter Total tasks couns: {len(tasks)}")
        
        import multiprocessing
        import random
        random.shuffle(tasks)

        num_workers = 5
        processes = []
        base_task_count = len(tasks) // num_workers
        extra_tasks = len(tasks) % num_workers
        start_index = 0

        print(f"Num workers: {num_workers}")
        print(f"base_task_count: {base_task_count}, extra_tasks: {extra_tasks}")

        end_index = -1
        self.evaluate_and_log(tasks[start_index:end_index], 1)

        # 使用 Manager 来共享进度数据
        # ctx = multiprocessing.get_context('spawn')
        # manager = ctx.Manager()
        # # 创建 spawn 上下文
        # for i in range(num_workers):
        #     end_index = start_index + base_task_count + (1 if i < extra_tasks else 0)
        #     process = ctx.Process(target=self.evaluate_and_log, args=(tasks[start_index:end_index], i))
        #     start_index = end_index
        #     processes.append(process)
        #     process.start()  # 启动进程

        # # 等待所有进程完成
        # for process in processes:
        #     process.join()  

    def evaluate_and_log(self, tasks, i):
        needle_dir = self.needle_dir
        needle_paths = [ os.path.join(needle_dir, needle_name) for needle_name in self.needle]
        
        if self.model_name == 'videoxl':  
            model_to_test = Beacon({"model_path": self.model_path, "device": i,
            'needle_paths':needle_paths, 'beacon_ratio':self.overwrite_beacon_ratio})
        elif self.model_name == 'other':  
            pass
        
        model_to_test.init_select_frames_tensors(self.context_lengths, self.pre_selecte_frames)

        for context_length, depth_percent in tqdm(tasks):

            avg_score = 0
            all_response = []
            unique_id = f'{context_length}_{depth_percent}'
            if self.load_gt_chunk:
                gt_chunk_idx = [gt_chunks_by_unique_id[unique_id]]
            else:
                gt_chunk_idx = None

            test_start_time = time.time()
            
            for idx, (retrieval_question, answer) in enumerate(zip(self.retrieval_question, self.answer)):
                
                # Prepare your message to send to the model you're going to evaluate
                prompt = model_to_test.generate_prompt(retrieval_question)    

                # Go see if the model can answer the question to pull out your random fact
                # try:
                response_start_time = time.time()
                response = model_to_test.generate(instruction=prompt, context_length=context_length,
                depth_percent=depth_percent, needle_idx=idx, gt_chunk_idx=gt_chunk_idx)   
                response_end_time = time.time()
                responsetime = response_end_time - response_start_time
                print("----------------")
                print(f'context_length:{context_length}')
                print(f'depth_percent:{depth_percent}')
                print(f'response: {response}')
                print(f'answer: {answer}')
                print("----------------")

                new_response = response.split(')')[0]
                if answer in new_response: 
                    score=1
                else:
                    score=0

                avg_score = avg_score + score
                all_response.append(response)

            avg_score = avg_score/len(self.answer)

            test_end_time = time.time()
            test_elapsed_time = test_end_time - test_start_time
                # Compare the reponse to the actual needle you placed
                # score = self.evaluation_model.evaluate_response(response)
            results = {
                # 'context' : context, # Uncomment this line if you'd like to save the context the model was asked to retrieve from. Warning: This will become very large.
                'model' : model_to_test.model_name,
                'retrieval_question': self.retrieval_question,
                'answer': self.answer,
                'context_length' : int(context_length),
                'depth_percent' : float(depth_percent),
                'gt_chunk_idx': gt_chunk_idx,
                'version' : self.results_version,
                'needle' : self.needle,
                'model_response' : all_response,
                'score' : avg_score,
                'test_duration_seconds' : test_elapsed_time,
                'test_timestamp_utc' : datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S%z')
            }

            self.testing_results.append(results)

            model_file_location = self.model_file_location
            context_file_location = f'{model_file_location}_modality_{self.needle_modality}_len_{context_length}_depth_{int(depth_percent)}'

            if self.print_ongoing_status:
                print (f"-- Test Summary -- ")
                print (f"Duration: {test_elapsed_time:.1f} seconds")
                print (f"Context: {context_length} seconds")
                print (f"Depth: {depth_percent}%")
                print (f"Score: {avg_score}")
                print (f"Response: {all_response}\n")

            model_file_location = self.model_file_location

            if self.save_results:
                # Save the context to file for retesting
                result_dir = self.result_dir
                result_dir = os.path.join(result_dir, model_file_location)
                if not os.path.exists(result_dir):
                    os.makedirs(result_dir)

                # Save the result to file for retesting
                with open(f'{result_dir}/{context_file_location}_results.json', 'w') as f:
                    json.dump(results, f)

    def print_start_test_summary(self):
        print ("\n")
        print ("Starting Needle In A Haystack Testing...")
        print (f"- Model: {self.model_name}")
        print (f"- Video Lengths: {len(self.context_lengths)}, Min: {min(self.context_lengths)}, Max: {max(self.context_lengths)}")
        print (f"- Video Depths: {len(self.video_depth_percents)}, Min: {min(self.video_depth_percents)}%, Max: {max(self.video_depth_percents)}%")
        # print (f"- Needle: {self.needle.strip()}: {self.needle_desc}")
        print (f"- Needle: {self.needle}: {self.needle_desc}")
        print ("\n\n")

    def start_test(self):
        if self.print_ongoing_status:
            self.print_start_test_summary()
        self.run_test()

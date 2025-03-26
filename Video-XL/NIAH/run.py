from dataclasses import dataclass, field
from typing import Optional, List
from jsonargparse import CLI
import os
from llm_needle_haystack_tester import LLMNeedleHaystackTester
import json
import pdb


@dataclass
class CommandArgs():
    load_gt_chunk: Optional[bool] = False
    overwrite_beacon_ratio: Optional[int] = 16
    model_name: str = "GPT4O"
    model_file_location: str = ""
    model_path: str = ""
    result_dir: str = ""
    retrieval_question: list[str] = field(default_factory=lambda: [])
    answer: list[str] = field(default_factory=lambda: ["D", "C", "C", "B", "A"])
    needle: list[str] = field(default_factory=lambda: ["readingsky.png", "_2dHBPpZAEw (online-video-cutter.com).png", "-errNPiGXF4 (online-video-cutter.com).png", "zIudsNnuaSY (online-video-cutter.com).png", "Zr6VEZQt9uQ (online-video-cutter.com).png"])
    needle_paths: list[str] = field(default_factory=lambda: [])
    needle_qa_path : Optional[str] = ''
    needle_desc: Optional[str] = "the young man seated on a cloud in the sky is reading a book"
    # needle_desc: Optional[str] = "reading a book"
    needle_modality: Optional[str] = "image"
    needle_dir: Optional[str] = "haystack"
    haystack_path: Optional[str] = ""
    results_version: Optional[int] = 1
    context_lengths_min: Optional[int] = 1
    context_lengths_max: Optional[int] = 320
    context_lengths_num_intervals: Optional[int] = 40
    context_lengths: Optional[list[int]] = None
    video_depth_percent_min: Optional[int] = 0
    video_depth_percent_max: Optional[int] = 100
    video_depth_percent_intervals: Optional[int] = 12
    video_depth_percents: Optional[list[int]] = None
    video_depth_percent_interval_type: Optional[str] = "linear"
    save_results: Optional[bool] = True
    final_context_length_buffer: Optional[int] = 200
    print_ongoing_status: Optional[bool] = True
    # Multi-needle parameters
    multi_needle: Optional[bool] = False # TODO: optimize multi_needle in video

def main():
    """
    The main function to execute the testing process based on command line arguments.
    
    It parses the command line arguments, selects the appropriate model model_name and evaluator,
    and initiates the testing process either for single-needle or multi-needle scenarios.
    """
    args = CLI(CommandArgs, as_positional=False)

    with open(args.needle_qa_path, 'r') as file:
        tmp_data = json.load(file)

    print(f'json needle_qa: {tmp_data}')

    args.retrieval_question = tmp_data['retrieval_question']
    args.answer = tmp_data['answer']
    args.needle = tmp_data['needle']

    args.needle_paths = [ os.path.join(args.needle_dir, needle_name) for needle_name in args.needle]
    print(args.needle_paths)
    args.evaluator = None
        
    if args.multi_needle == True:
        print("Testing multi-needle")
        tester = LLMMultiNeedleHaystackTester(**args.__dict__)
    else: 
        print("Testing single-needle")
        tester = LLMNeedleHaystackTester(**args.__dict__)

    tester.start_test()

if __name__ == "__main__":
    main()

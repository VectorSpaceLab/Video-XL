import argparse
import pdb
import os
import json

def get_args():
    parser = argparse.ArgumentParser(
        description="Merge results from different sub-task directories."
    )
    parser.add_argument("--benchmark", type=str, default="videomme")
    parser.add_argument(
        "--results_dir_list",
        nargs='+', 
        type=str,
        required=True,
        help="A space-separated list of directories containing the results to merge."
    )
    parser.add_argument("--save_dir", type=str, default="./")
    args = parser.parse_args()
    return args

def load_json(file_path):
    return json.load(open(file_path, "r", encoding="utf-8"))

def videomme_merge(results_dir_list, save_dir):
    merge_results = []
    for results_dir in results_dir_list:
        # load json file
        sub_dir = os.listdir(results_dir)[0]
        sub_dir = os.path.join(results_dir, sub_dir)
        json_file_names = os.listdir(sub_dir)
        target_file_name = [name for name in json_file_names if 'videomme' in name][0]
        target_file_path = os.path.join(sub_dir, target_file_name)
        sub_results = load_json(target_file_path)['logs']
        merge_results.extend(sub_results)

    score_by_task_type = {}
    for res in merge_results:
        result_info = res['videomme_percetion_score']
        task_type = result_info['task_category']
        if task_type not in score_by_task_type:
            score_by_task_type[task_type] = {'correct': 0, 'total': 0}
        
        score_by_task_type[task_type]['total'] += 1
        score_by_task_type[task_type]['correct'] += result_info['pred_answer'] == result_info['answer']

    # logging:
    global_total = 0
    global_correct = 0

    print("\n" + "=" * 60)
    print(f"{'ACCURACY REPORT'.center(60)}")
    print("=" * 60)

    # Print table header
    print(f"{'Task Type'.ljust(20)} | {'Total'.ljust(8)} | {'Correct'.ljust(8)} | {'Accuracy'.ljust(10)}")
    print("-" * 60)

    for task_type, info in score_by_task_type.items():
        task_accuracy = (info['correct'] / info['total']) * 100 if info['total'] > 0 else 0
        print(f"{task_type.ljust(20)} | {str(info['total']).ljust(8)} | {str(info['correct']).ljust(8)} | {task_accuracy:.2f}%".ljust(10))
        score_by_task_type[task_type]['accuracy'] = task_accuracy
        global_total += info['total']
        global_correct += info['correct']

    print("=" * 60)
    global_accuracy = (global_correct / global_total) * 100 if global_total > 0 else 0
    print(f"{'GLOBAL ACCURACY'.ljust(20)} | {str(global_total).ljust(8)} | {str(global_correct).ljust(8)} | {global_accuracy:.2f}%".ljust(10))
    print("=" * 60 + "\n")
    score_by_task_type['global'] = {
        'total': global_total,
        'correct': global_correct,
        'accuracy': global_accuracy
    }

    # save 
    complete_save_dir = os.path.join(save_dir, 'videomme_merge')
    os.makedirs(complete_save_dir, exist_ok=True)
    save_path = os.path.join(complete_save_dir, 'videomme.json')
    with open(save_path, 'w') as f:
        json.dump(score_by_task_type, f, indent=4)
    print(f'Merge Results Saved to {save_path}')



def lvbench_merge(results_dir_list, save_dir):
    merge_results = []
    for results_dir in results_dir_list:
        # load json file
        sub_dir = os.listdir(results_dir)[0]
        sub_dir = os.path.join(results_dir, sub_dir)
        json_file_names = os.listdir(sub_dir)
        target_file_name = [name for name in json_file_names if 'lvbench' in name][0]
        target_file_path = os.path.join(sub_dir, target_file_name)
        sub_results = load_json(target_file_path)['logs']
        merge_results.extend(sub_results)

    score_by_task_type = {}
    for res in merge_results:
        result_info = res['lvbench_mc_accuracy']
        task_type = res['doc']['question_type'][0]
        if task_type not in score_by_task_type:
            score_by_task_type[task_type] = {'correct': 0, 'total': 0}
        
        score_by_task_type[task_type]['total'] += 1
        score_by_task_type[task_type]['correct'] += result_info['pred_answer'] == result_info['gt_answer']

    # logging:
    global_total = 0
    global_correct = 0

    print("\n" + "=" * 60)
    print(f"{'ACCURACY REPORT'.center(60)}")
    print("=" * 60)

    # Print table header
    print(f"{'Task Type'.ljust(20)} | {'Total'.ljust(8)} | {'Correct'.ljust(8)} | {'Accuracy'.ljust(10)}")
    print("-" * 60)

    for task_type, info in score_by_task_type.items():
        task_accuracy = (info['correct'] / info['total']) * 100 if info['total'] > 0 else 0
        print(f"{task_type.ljust(20)} | {str(info['total']).ljust(8)} | {str(info['correct']).ljust(8)} | {task_accuracy:.2f}%".ljust(10))
        score_by_task_type[task_type]['accuracy'] = task_accuracy
        global_total += info['total']
        global_correct += info['correct']

    print("=" * 60)
    global_accuracy = (global_correct / global_total) * 100 if global_total > 0 else 0
    print(f"{'GLOBAL ACCURACY'.ljust(20)} | {str(global_total).ljust(8)} | {str(global_correct).ljust(8)} | {global_accuracy:.2f}%".ljust(10))
    print("=" * 60 + "\n")
    score_by_task_type['global'] = {
        'total': global_total,
        'correct': global_correct,
        'accuracy': global_accuracy
    }

    # save 
    complete_save_dir = os.path.join(save_dir, 'lvbench_merge')
    os.makedirs(complete_save_dir, exist_ok=True)
    save_path = os.path.join(complete_save_dir, 'lvbench.json')
    with open(save_path, 'w') as f:
        json.dump(score_by_task_type, f, indent=4)
    print(f'Merge Results Saved to {save_path}')



if __name__ == "__main__":
    args = get_args()
    results_dir_list = args.results_dir_list
    for results_dir in results_dir_list:
        print(f'results need to merge:')
        print(f'- "{results_dir}"')

    save_dir = args.save_dir
    benchmark = args.benchmark
    if benchmark == "videomme":
        videomme_merge(results_dir_list, save_dir)
    elif benchmark == "lvbench":
        lvbench_merge(results_dir_list, save_dir)
    
    

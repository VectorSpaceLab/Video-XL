source /opt/conda/bin/activate /share/minghao/Envs/videoxl_train
cd /share/minghao/Projects/NIAH
start_time=$(date +%s)

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u run.py \
    --overwrite_beacon_ratio 16 \
    --model_name videoxl \
    --context_lengths_min 1 \
    --context_lengths_max 2048 \
    --context_lengths_num_intervals 12 \
    --video_depth_percent_min 0 \
    --video_depth_percent_max 100 \
    --video_depth_percent_intervals 5 \
    --model_file_location videoxl \
    --model_path "/share/shuyan/VideoXL_weight_8" \
    --result_dir "/share/minghao/Projects/NIAH/results" \
    --haystack_path /share/junjie/shuyan/NeedleInAVideoHaystack/needlehaystack/haystack/long_other/203.mp4 \
    --needle_qa_path /share/minghao/Projects/NIAH/datas/qa/qa.json \
    --needle_dir /share/minghao/Projects/NIAH/datas/needles 2>&1 | tee ./log/videoxl.log

# print runing time
end_time=$(date +%s)
elapsed_time=$(( end_time - start_time ))
echo "cost: $elapsed_time s"

# visualize the image
python vis.py \
 --model_name videoxl \
 --results_dir /share/minghao/Projects/NIAH/results

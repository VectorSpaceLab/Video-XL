#!/bin/bash
# For example, if Video-XL is in /home/user/my_projects/Video-XL, set this to /home/user/my_projects/Video-XL
PROJECT_ROOT="/path/to/your/Video-XL-2_project_root" # <--- **REQUIRED: CHANGE THIS PATH**

# Path to your conda environment for Video-XL-2
CONDA_ENV_PATH="/path/to/your/conda/env/xx" # <--- **REQUIRED: CHANGE THIS PATH**

# Path to the pre-trained Video-XL-2 model
MODEL_PATH="${PROJECT_ROOT}/models/Video-XL-2" # <--- **RECOMMENDED: PLACE YOUR MODEL HERE, OR CHANGE THIS PATH**
# Alternatively, if your model is elsewhere:

# --- Hugging Face Configuration (Optional, for mirror or token access) ---
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}" # Default to hf-mirror, can be overridden by user
export HF_TOKEN="${HF_TOKEN:-}" # Optional: set your HF token if needed for private models or higher rate limits

# --- Script Execution ---

echo "--- Starting Video-XL-2 Evaluation Script ---"
echo "Project Root: ${PROJECT_ROOT}"
echo "Conda Environment: ${CONDA_ENV_PATH}"
echo "Model Path: ${MODEL_PATH}"
echo "HF Endpoint: ${HF_ENDPOINT}"

# Change to the evaluation directory
# This assumes the script is run from or symlinked to PROJECT_ROOT/Video-XL-2/eval/lvu/w_chunk_bilevel
cd "${PROJECT_ROOT}/Video-XL-2/eval/lvu/w_chunk_bilevel"

# Activate the specified conda environment
echo "Activating conda environment: ${CONDA_ENV_PATH}"
source "${CONDA_ENV_PATH}/bin/activate" || { echo "Error: Could not activate conda environment. Please check CONDA_ENV_PATH."; exit 1; }

# Run the evaluation
echo "Running accelerate launch command..."
for task_type in short medium long; do
    echo "Processing task type: ${task_type^^}..."
    accelerate launch --num_processes 8 --main_process_port 12345 -m lmms_eval \
        --model videoxl2 \
        --model_args "pretrained=$MODEL_PATH,conv_template=qwen_1_5,model_name=llava_qwen,max_frames_num=400,fps=1,max_fps=4,block_size_chosed=4,prev_blocks_num=3,video_decode_backend=decord,selected_info_file_path=./selected_infos/videomme_${task_type}.json,attn_implementation=sdpa," \
        --tasks "videomme_${task_type}" \
        --batch_size 1 \
        --log_samples \
        --log_samples_suffix videoxl2 \
        --output_path "./logs/videoxl2-w_chunk_bi-videomme_${task_type}" 2>&1 | tee "./printlogs/videoxl2-w_chunk_bi-videomme_${task_type}.log"

    if [ $? -eq 0 ]; then
        echo "  Successfully completed evaluation for ${task_type^^}."
    else
        echo "  ERROR: Evaluation for ${task_type^^} failed. Check logs for details."
        exit 1 # Exit script if any sub-task fails
    fi
    echo "---------------------------------------------------"

done

echo "All individual evaluations completed."
echo "Starting results merging process..."

python merge_results.py \
    --benchmark videomme \
    --results_dir_list \
        ./logs/videoxl2-w_chunk_bi-videomme_short \
        ./logs/videoxl2-w_chunk_bi-videomme_medium \
        ./logs/videoxl2-w_chunk_bi-videomme_long \
    --save_dir ./logs

# Deactivate conda environment (optional, but good practice)
# conda deactivate
echo "--- Evaluation Script Finished ---"
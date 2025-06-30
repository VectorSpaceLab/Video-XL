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
# This assumes the script is run from or symlinked to PROJECT_ROOT/Video-XL-2/eval/lvu/pure
cd "${PROJECT_ROOT}/Video-XL-2/eval/lvu/pure"

# Activate the specified conda environment
echo "Activating conda environment: ${CONDA_ENV_PATH}"
source "${CONDA_ENV_PATH}/bin/activate" || { echo "Error: Could not activate conda environment. Please check CONDA_ENV_PATH."; exit 1; }

# Run the evaluation
echo "Running accelerate launch command..."
accelerate launch \
    --num_processes 8 \
    --main_process_port 12345 \
    -m lmms_eval \
    --model videoxl2 \
    --model_args "pretrained=${MODEL_PATH},conv_template=qwen_1_5,model_name=llava_qwen,max_frames_num=400,fps=1,max_fps=4,video_decode_backend=decord" \
    --tasks videomme \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix videoxl2 \
    --output_path ./logs/videoxl2-videomme 2>&1 | tee ./printlogs/videoxl2-videomme.log

# Deactivate conda environment (optional, but good practice)
# conda deactivate

echo "--- Evaluation Script Finished ---"
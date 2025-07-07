#!/bin/bash
# This script trains the Video-XL-2 model with specified configurations.

# --- Configuration Variables ---
# You can modify these paths and settings as needed.
# Base directory for the Video-XL-2 project
PROJECT_DIR="/root/Video-XL/Video-XL-2"

# Path to the Conda environment for Video-XL
CONDA_ENV_PATH="YOUR_CONDA_ENV_PATH"

# Path to the Qwen model
BASE_MODEL_PATH="/Root/Model/Qwen2.5-7B-Instruct"

# Path to the vision tower model (SigLIP)
VISION_TOWER_PATH="/Root/Model/siglip-so400m-patch14-384"

# Path to the pre-trained MLP adapter
MLP_PROJECTOR_PATH="/Root/Model/VideoXL2_Stage2/pretrain_mlp_projector.bin"

# Path to the pre-trained DTS encoder
DTS_PATH="/Root/Model/VideoXL2_Stage1/dts.pth"

# Path to the pre-trained DTS encoder
IMAGE_FOLDER="/Root/Datasets/ImageDatas/"
VIDEO_FOLDER="/Root/Datasets/VideoDatas/"

# Output directory for checkpoints
OUTPUT_DIR="./checkpoints/videoxl2_stage3"

# Data configuration file
DATA_PATH="./datas/stage3_datas.yaml"

# DeepSpeed configuration file
DEEPSPEED_CONFIG="./deepspeed/zero1.json"

# --- Hardware Configuration ---
ARG_WORLD_SIZE=${1:-1}
ARG_NPROC_PER_NODE=${2:-8}
ARG_MASTER_ADDR="127.0.0.1"
ARG_MASTER_PORT=16667
ARG_RANK=0

# Multiple conditions
if [ ! -n "$WORLD_SIZE" ] || [ ! -n "$NPROC_PER_NODE" ]; then
    WORLD_SIZE=$ARG_WORLD_SIZE
    NPROC_PER_NODE=$ARG_NPROC_PER_NODE
fi
if [ ! -n "$MASTER_ADDR" ] || [ ! -n "$MASTER_PORT" ] || [ ! -n "$RANK" ]; then
    MASTER_ADDR=$ARG_MASTER_ADDR
    MASTER_PORT=$ARG_MASTER_PORT
    RANK=$ARG_RANK
fi

echo "WORLD_SIZE: $WORLD_SIZE"
echo "NPROC_PER_NODE: $NPROC_PER_NODE"
echo "ARG_MASTER_ADDR: $ARG_MASTER_ADDR"
echo "ARG_MASTER_PORT: $ARG_MASTER_PORT"
echo "RANK: $RANK"

# --- Environment Setup ---
echo "Changing directory to $PROJECT_DIR/train..."
cd "$PROJECT_DIR/train" || { echo "Failed to change directory."; exit 1; }

echo "Activating Conda environment from $CONDA_ENV_PATH..."
source activate "$CONDA_ENV_PATH" || { echo "Failed to activate conda environment."; exit 1; }

# Set environment variables for multi-threading
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NCCL_DEBUG=WARN # Set to INFO for more detailed NCCL debugging

# --- Start Training ---
echo "Starting distributed training with torchrun..."
torchrun \
    --nnodes="$WORLD_SIZE" \
    --nproc_per_node="$NPROC_PER_NODE" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    --node_rank="$RANK" \
    videoxl2/train/train_mem.py \
    --deepspeed "$DEEPSPEED_CONFIG" \
    --model_name_or_path "$BASE_MODEL_PATH" \
    --version qwen_1_5 \
    --data_path "$DATA_PATH" \
    --image_folder "$IMAGE_FOLDER" \
    --video_folder "$VIDEO_FOLDER" \
    --vision_tower "$VISION_TOWER_PATH" \
    --pretrain_mm_mlp_adapter "$MLP_PROJECTOR_PATH" \
    --pretrain_dts "$DTS_PATH" \
    --mm_projector_type mlp2x_gelu \
    --video_fps 1 \
    --frames_upbound 16 \
    --image_aspect_ratio anyres \
    --group_by_modality_length True \
    --unfreeze_mm_vision_tower True \
    --mm_spatial_pool_stride 2 \
    --mm_resampler_type "spatial_pool" \
    --mm_spatial_pool_out_channels 1152 \
    --mm_vision_select_feature patch \
    --image_grid_pinpoints "(1x1)...(6x6)" \
    --mm_patch_merge_type unires \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 1.0e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --run_name finetune_stage3 \
    --group_by_stride strict

echo "Training script finished."
# Train Video-XL-2

## 1. Setup
```bash
git clone https://github.com/VectorSpaceLab/Video-XL
cd Video-XL-2
pip install -e .
```

## 2. Training Data
We provide the training data infomations for Video-XL-2 in [./datas](./datas).

## 3. Training
<!-- | Stage | Num. frames | ViT | Connector | LLM | CKPT |
|--------|:-------:|:------:|:------:|:------:|:------:|
| [stage1](scripts/train/stage1-init_connector) | 4 | :snowflake: | :fire: | :snowflake: | [all projector weights](https://huggingface.co/OpenGVLab/stage1-mm-projectors/tree/main) |
| [stage2](scripts/train/stage2-visual_pretraining) | 4-8 | :fire: | :fire: | :fire: | [UMT-Qwen2_7B](https://huggingface.co/OpenGVLab/stage2-UMT-Qwen2-7B-tome16_mlp), [UMT-Qwen2_5_1M_7B](https://huggingface.co/OpenGVLab/stage2-UMT-Qwen2_5_7B_1m-tome16_mlp), [UMT-HD-Qwen2_5_2B](https://huggingface.co/OpenGVLab/stage2-UMT-Qwen2_5_1.5B-tome16_mlp), [InternVideo2-Qwen2_5_7B](https://huggingface.co/OpenGVLab/stage2-InternVideo2-1B-Qwen2_5-7B-tome16_mlp) |
| [stage3](scripts/train/stage3-video_sft) | 64-512 | :fire: | :fire: | :fire: | [UMT-Qwen2_7B](https://huggingface.co/OpenGVLab/VideoChat-Flash-Qwen2-7B_res448),[UMT-HD-Qwen2_5-2B](https://huggingface.co/OpenGVLab/VideoChat-Flash-Qwen2_5-2B_res448),[UMT-Qwen2_5_1M_7B](https://huggingface.co/OpenGVLab/VideoChat-Flash-Qwen2_5-7B-1M_res224), [InternVideo2-Qwen2_5_7B](https://huggingface.co/OpenGVLab/VideoChat-Flash-Qwen2_5-7B_InternVideo2-1B) |
| [stage4](scripts/train/stage4_highres_postft) | 64-512 | :fire: | :fire: | :snowflake: | [UMT-HD-Qwen2-7B](https://huggingface.co/OpenGVLab/VideoChat-Flash-Qwen2-7B_res448)| -->

<!-- Training time with a 32 A100:
- stage1: under one hour:
- stage2: about 2 day
- stage3: about 2~3day
- stage4: about 2~3day -->

### Stage-1: DTS Pre-training
```bash
```
### Stage-2: Visual-Language Alignment
```bash
```
### Stage-3: Visual Pre-training
```bash
```

### Stage-4: Visual SFT
```bash
```

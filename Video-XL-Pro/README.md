<p align="center">
    <img src="./assets/logo.jpg" width="100">
</p>



## Video-XL-Pro: Reconstructive Token Compression for Extremely Long Video Understanding
<!-- <p align="center">
    🌐 <a href="https://www.xiaohongshu.com/discovery/item/67172f5d0000000024017704?source=webshare&xhsshare=pc_web&xsec_token=GBL17lee3zbjumPCcki1x6IL0okkah9Lp3XX_IzlJwO4I=&xsec_source=pc_share" target="_blank">Blog</a> | 📃 <a href="https://arxiv.org/pdf/2409.14485" target="_blank">Paper</a> | 🤗 <a href="https://huggingface.co/sy1998/Video_XL" target="_blank">Model</a> |  🤗 <a href="https://huggingface.co/datasets/sy1998/Video_XL_Training/tree/main" target="_blank">Data</a> |  🎥 <a href="" target="_blank">Demo</a>

</p> -->

<p align="center">
    <img src="./assets/needle.png" width="800">
</p>
<p align="center"><em>(Left) The performance and max frames of different models.<br>(Right) Results on Needle-in-a-haystack evaluation on a single 80G GPU.
    </em></p>



✨ **Highlights**:

(i) Comprehensive long video understanding. Video-XL-Pro 3B achieves the **leading performance among 3B models** on MLVU, VideoMME, VNBench and LongVideoBench.

(ii) Efficient Long visual context processing. Video-XL-Pro can process **10000 frames on an 80G GPU and achieves nearly 98% accuracy** on Needle-in-a-haystack evaluation.



## Model weights
Please download our pre-trained and finetuned model weights from the [link](https://huggingface.co/lxr2003/Video-XL-Pro-3B) 
  
## Installation 
```bash
conda create -n videoxlpro python=3.10 -y && conda activate videoxlpro
pip install torch==2.1.2 torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -e "videoxlpro/.[train]"
pip install packaging &&  pip install ninja && pip install flash-attn --no-build-isolation --no-cache-dir
pip install -r requirements.txt
```

## Quick Start With HuggingFace

<details>
    <summary>Example Code</summary>
    
```python
import torch
import transformers
import gc
from videoxlpro.videoxlpro.demo_utils import process_video, load_image_processor, generate_response
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings

# 禁用一些警告
transformers.logging.set_verbosity_error()
warnings.filterwarnings('ignore')

# 设置设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 模型路径
model_path = "/path/to/your/Video-XL-Pro-3B"
video_path = "/path/to/your/video.mp4"

# 使用 Auto 类加载模型
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    low_cpu_mem_usage=True, 
    torch_dtype=torch.float16,
    attn_implementation="flash_attention_2",
    device_map=device,
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True
)

image_processor = load_image_processor(model, tokenizer)

max_frames_num = 128

# 处理视频
video_tensor = process_video(video_path, image_processor, model.device, max_frames_num)

# 生成参数
gen_kwargs = {
    "do_sample": True,
    "temperature": 0.01,
    "top_p": 0.001,
    "num_beams": 1,
    "use_cache": True,
    "max_new_tokens": 256
}

# 文本提示
prompt = "Describe this video."

text = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<image>\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

response = generate_response(model, tokenizer, text, video_tensor, gen_kwargs)

# 4. 输出结果
print("\n===== 生成的回答 =====")
print(response)
```
</details>

## Pre-training 
```bash
bash scripts/pretrain.sh
```

## Fine-tuning

```bash
bash scripts/finetune_v.sh
```

## Long Video Benchmark Evaluation

<details>
<summary>Expand to see the performance on Video-MME and MLVU</summary>
<IMG src="./assets/videomme.png"/>
</details>



## License
This project utilizes certain datasets and checkpoints that are subject to their respective original licenses. Users must comply with all terms and conditions of these original licenses.
The content of this project itself is licensed under the [Apache license 2.0](./LICENSE).





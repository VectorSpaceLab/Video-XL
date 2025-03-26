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

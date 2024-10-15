# Video-XL: Extra-Long Vision Language Model for Hour-Scale Video Understanding
<p align="center">
    <img src="./assets/needle.png" width="800">
</p>
<p align="center"><em>Results on Needle-in-a-haystack evaluation on a single 80G GPU.</em></p>
<p align="center">
    🌐 <a href="" target="_blank">Blog</a> | 📃 <a href="" target="_blank">Paper</a> | 🤗 <a href="" target="_blank">Hugging Face</a> | 🎥 <a href="" target="_blank">Demo</a>

</p>



Video-XL is an extra-long vision language model for hour-scale video understanding. With LLM compression, Video-XL can easily extend VLM to longer visual contexts wihout inforamtion loss. 

✨ **Highlights**:

(i) Comprehensive long video understanding. Video-XL 7B achieves the leading performance among 7B models on MLVU, VideoMME, VNBench and LongVideoBench.

(ii) Efficient Long visual context processing. Video-XL can process 2048 frames on an 80G GPU and achieves nearly 95% accuracy on Needle-in-a-haystack evaluation.

(iii) Video-XL shows strong ability in some real-world scenarios, like movie summarization, surveillance anomaly detection and Ad placement identification.



## News
- [2024/10/15] 🔥 Video-XL is released,  including model, training and evaluation code. 
  
## Installation 
```bash
conda create -n videoxl python=3.10 -y && conda activate videoxl
pip install torch==2.1.2 torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -e "videoxl/.[train]"
pip install packaging &&  pip install ninja && pip install flash-attn --no-build-isolation --no-cache-dir
pip install -r requirements.txt
```


## Plan

 - [ ] Technical Report
 - [ ] Model
 - [ ] Code
 - [ ] Data



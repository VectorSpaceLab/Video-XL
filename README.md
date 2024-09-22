<p align="center">
    <img src="https://github.com/VectorSpaceLab/XL-VLM/blob/main/assets/logo.jpg" width="200" style="margin-bottom: 0.2;"/>
</p>

<h3 align="center" style="font-size: 50px;">
    <a style="color:#9C276A;">
        XL-VLM: Extra-Long Vision Language Model for Hour-Scale Video Understanding
    </a>
</h3>
<h5 align="center"> If our project helps you, please give us a star ⭐ on GitHub to support us. 🙏🙏 </h5>

## Overview
XL-VLM is an extra-long vision language model for hour-scale video understanding. With LLM compression, XL-VLM can easily extend VLM to longer visual contexts wihout inforamtion loss. 
✨ Highlights:
(i) Comprehensive long video understanding. XL-VLM 7B achieves the leading performance among 7B models on MLVU, VideoMME, VNBench and LongVideoBench.
(ii) Efficient Long visual context processing. XL-VLM can process 1024 frames on an 80G GPU and achieves 100% accuracy on Needle-in-a-haystack evaluation.
(iii) XL-VLM shows strong ability in some real-world scenarios, like video summarization, surveillance anomaly detection and Ad placement identification.

## Plan

 - [ ] Technical Report
 - [ ] Model
 - [ ] Code
 - [ ] Data



VideoLLaMB is a novel long video comprehension framework utilizing Memory Bridge Layers with recurrent memory tokens to encode 100% video content without discarding critical visual cues.

✨ Highlights:

Comprehensive long video understanding. VideoLLaMB-7B reached the state-of-the-art performance among 7B models trained on vicuna-7b and videochat2 video on EgoSchema, NexTQA and MVBench, reaching 8x longer video length with robust performance in comparison to PLLaVA.

Memory-based egocentric planning. VideoLLaMB achieves the best performance among all video-language models on EgoPlan, with an improvement of  over PLLaVA.

Training-free streaming captioning. With our SceneTiling algorithm, VideoLLaMB can capture the dynamics with in a streaming video and directly predict the streaming captions in real-time, without the need to process the entire video sequence beforehand.

Enhanced frame retrieval on needle in a video haystack (NIAVH). We present the “Needle in a Video Haystack” (NIAVH) benchmark to evaluate long video understanding over needle of different modalities comprehensively (details 👉). In the pressure test ranging from 1 to 300 seconds in length, VideoLLaMB consistently retrieves the correct image needles at various depths, outperforming other methods as video length increases

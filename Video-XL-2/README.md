<div align='center'>
<h1>Video-XL-2: Towards Very Long-Video Understanding Through Task-Aware KV Sparsification </h1>
<h3></h3>

| [Blog](https://unabletousegit.github.io/video-xl2.github.io/) | [Tech Report](https://arxiv.org/abs/2506.19225) | [🤗HF Models](https://huggingface.co/BAAI/Video-XL-2) |

</div>


<div align='center'>
<img src="./assets/performance.png" class="interpolation-image" alt="arch." height="80%" width="70%" />
</div>

We introduce **Video-XL-2**, a new suite of multimodal models that achieves state-of-the-art (SOTA) performance and superior efficiency in long video understanding.

### Video-XL-2: SOTA Performance and Unrivaled Efficiency
**Video-XL-2** achieves SOTA performance in mainstream long video understanding benchmarks and leading performance in temporal grounding tasks when compared to open-source lightweight models. Furthermore, it boasts significant advantages over existing models in both memory consumption and inference speed."
<!-- 
<div align='center'>
<img src="./assets/comparison.png" class="interpolation-image" alt="comparison." height="80%" width="80%" />
</div> -->
<!-- 
### Highlights
- **Emu3** is capable of generating high-quality images following the text input, by simply predicting the next vision token. The model naturally supports flexible resolutions and styles.
- **Emu3** shows strong vision-language understanding capabilities to see the physical world and provides coherent text responses. Notably, this capability is achieved without depending on a CLIP and a pretrained LLM.
- **Emu3** simply generates a video causally by predicting the next token in a video sequence, unlike the video diffusion model as in Sora. With a video in context, Emu3 can also naturally extend the video and predict what will happen next.  -->


<!-- ### TODO
- [X] Release model weights.
- [ ] Release the inference code.
- [X] Release the training code for sft.
- [ ] Release the training guidance.
- [X] Release the evaluation code.
- [ ] Release the evaluation guidance. -->


### Model Weights

| Model name| HF Weight |
| ------------------------ | -------------------------------------------------------------- | 
| **Video-XL-2/Stage1**          | [🤗 HF link](https://huggingface.co/BAAI/Stage1_and_Stage2_Weights)  |
| **Video-XL-2/Stage2**           | [🤗 HF link](https://huggingface.co/BAAI/Stage1_and_Stage2_Weights) |
| **Video-XL-2/Stage3**          | [🤗 HF link](https://huggingface.co/BAAI/Stage3_Weights)  |
| **Video-XL-2/Stage4**           | [🤗 HF link](https://huggingface.co/BAAI/Video-XL-2) |

<!-- ### Quickstart -->

### Setup

Clone this repository and install required packages:

```shell
git clone https://github.com/VectorSpaceLab/Video-XL
cd Video-XL-2
pip install -r requirements.txt
```

<!-- #### Use 🤗Transformers to run Video-XL-2 for video understanding
```python
``` -->

### Training
The training codes and scripts can be found in [./train](./train).

### Evaluation
The evaluation codes and scripts can be found in [./eval](./eval).


## Acknowledgement
We thank the great work from [Video-XL Series](https://github.com/VectorSpaceLab/Video-XL), [LongVA](https://github.com/QwenLM/Qwen2-VL), [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval), [Qwen](https://github.com/QwenLM/Qwen),[VideoChat-Flash](https://github.com/OpenGVLab/VideoChat-Flash?tab=readme-ov-file).

## Citation

If you find Video-XL-2 useful for your research and applications, please consider starring this repository and citing:

```
@article{qin2025video,
  title={Video-XL-2: Towards Very Long-Video Understanding Through Task-Aware KV Sparsification},
  author={Qin, Minghao and Liu, Xiangrui and Liang, Zhengyang and Shu, Yan and Yuan, Huaying and Zhou, Juenjie and Xiao, Shitao and Zhao, Bo and Liu, Zheng},
  journal={arXiv preprint arXiv:2506.19225},
  year={2025}
}

@article{shu2024video,
  title={Video-XL: Extra-Long Vision Language Model for Hour-Scale Video Understanding},
  author={Shu, Yan and Zhang, Peitian and Liu, Zheng and Qin, Minghao and Zhou, Junjie and Huang, Tiejun and Zhao, Bo},
  journal={arXiv preprint arXiv:2409.14485},
  year={2024}
}

@article{liu2025video,
  title={Video-XL-Pro: Reconstructive Token Compression for Extremely Long Video Understanding},
  author={Liu, Xiangrui and Shu, Yan and Liu, Zheng and Li, Ao and Tian, Yang and Zhao, Bo},
  journal={arXiv preprint arXiv:2503.18478},
  year={2025}
}
```


# Train Video-XL-2
This sub repo provides the training code for **Video-XL-2**.


## 1. Installation
To get started, clone the repository and install the necessary dependencies.

```bash
git clone https://github.com/VectorSpaceLab/Video-XL
cd ./Video-XL/train
conda create -n video_xl_env python=3.10 # Create a new conda environment named 'video_xl_env'
conda activate video_xl_env             # Activate your environment
pip install -r requirements.txt
```

## 2. Training Data
The training data configuration files for **stage3** and **stage4** are located in `./configs/data`. These files contain source links to help you download the raw video and image data.

Their corresponding annotation JSON files can be downloaded directly from here (https://huggingface.co/datasets/BAAI/VideoXL2_Training_Data_Anno_Files)

**Important**: After downloading your video and image data,you'll need to keep all video and image datasets in their respective directories

For example, your directory structure should look like this:
```
/path/to/your/local/datas
├── videodatas
│   ├── datasets_1
│   ├── datasets_2
│   ├── datasets_3
│   ├── ...
├── imagedata
│   ├── datasets_4
│   ├── datasets_5
│   ├── datasets_6
│   ├── ...
```

```
## 3. Model Weights

Download the following pre-trained weights, which are essential for **stage3** and **stage4** fine-tuning.

  * **For Stage3 fine-tuning, you will need:**
      * **DTS module weight** (from stage1): [Download Link](https://huggingface.co/BAAI/Stage1_and_Stage2_Weights)
      * **MLP projector weight** (from stage2): [Download Link](https://huggingface.co/BAAI/Stage1_and_Stage2_Weights)
  * **For Stage4 fine-tuning, you will need:**
      * **Complete model weight** (from stage3): [Download Link](https://huggingface.co/BAAI/Video-XL-2-Stage3)


## 4. Training

Training for Stage1 and Stage2 is consistent with **Video-XL-Pro**. For details and training code related to these initial stages, please refer to the Video-XL-Pro repository.

For **stage3** and **stage4** training, use the following commands. The parameters `4` and `8` in the scripts represent the **number of machines** and the **number of GPUs per machine**, respectively.

```bash
cd ./Video-XL/train
bash ./scripts/train_stage3.sh 4 8 # Example: Train on 4 machines, each with 8 GPUs
bash ./scripts/train_stage4.sh 4 8 # Example: Train on 4 machines, each with 8 GPUs
```
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
The training data configuration files for **stage3** and **stage4** are located in `./configs/data`. These files contain source links to help you download the raw video data.

Their corresponding annotation JSON files can be downloaded directly from here (https://www.google.com/search?q=https://drive.google.com/

**Important**: After downloading your video data to a local directory (e.g., `/path/to/your/local/videos`), you'll need to update the `data_path` in the annotation JSON files to point to your local video storage. Run the following command, replacing `/path/to/your/local/videos` with the actual absolute path to your video dataset:

```bash
python ./preprocess/train.py --anno_file_path ./data/stage3_train.json --local_video_dir /path/to/your/local/videos
```


## 3. Model Weights

Download the following pre-trained weights, which are essential for **stage3** and **stage4** fine-tuning.

  * **For Stage3 fine-tuning, you will need:**
      * **DTS module weight** (from stage1): [Download Link](https://www.google.com/search?q=YOUR_DTS_MODULE_WEIGHT_LINK)
      * **MLP projector weight** (from stage2): [Download Link](https://www.google.com/search?q=YOUR_MLP_PROJECTOR_WEIGHT_LINK)
  * **For Stage4 fine-tuning, you will need:**
      * **Complete model weight** (from stage3): [Download Link](https://www.google.com/search?q=YOUR_COMPLETE_STAGE3_MODEL_LINK)


## 4. Training

Training for Stage1 and Stage2 is consistent with **Video-XL-Pro**. For details and training code related to these initial stages, please refer to the Video-XL-Pro repository.

For **stage3** and **stage4** training, use the following commands. The parameters `4` and `8` in the scripts represent the **number of machines** and the **number of GPUs per machine**, respectively.

```bash
cd ./Video-XL/train
bash ./scripts/train_stage3.sh 4 8 # Example: Train on 4 machines, each with 8 GPUs
bash ./scripts/train_stage4.sh 4 8 # Example: Train on 4 machines, each with 8 GPUs
```
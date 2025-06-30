## **1. Installation**

To get started, clone the repository and install the required dependencies.

```bash
git clone https://github.com/VectorSpaceLab/Video-XL
cd ./Video-XL/eval
conda activate your_conda_env # Replace 'your_conda_env' with your desired environment name
pip install -r requirements.txt
```

-----

## **2. Reproduce Evaluation Results**

This section provides a guide for reproducing the evaluation results for Video-XL-2. Before you begin, please ensure you have completed theInstallation steps.

### **2.1 Long Video Understanding (LVU)**

#### **Step 1: Prepare Models and Benchmarks**

1.  **Download our model** from [HF](https://huggingface.co/BAAI/Video-XL-2).
2.  **Download benchmark videos** from their original sources. You will need to update the videos' directory path in the `utils.py` file located in `lmms_eval/tasks/benchmark_name` for each benchmark to point to your local video directory.

#### **Step 2: Run Evaluation Scripts**

We provide three distinct evaluation settings: **Pure**, **w/ Chunk-based Pre-filling**, and **w/ Chunk-based Pre-filling + Bi-level Decoding**. Navigate to the respective directories and run the evaluation scripts for each setting.

  * **Setting 1: Pure Video-XL-2**

      * **Directory:** `./Video-XL-2/eval/lvu/pure/scripts`
      * This setting evaluates the base Video-XL-2 model without any efficiency optimizations.

  * **Setting 2: w/ Chunk-based Pre-filling**

      * **Directory:** `./Video-XL-2/eval/lvu/w_chunk/scripts`
      * This setting evaluates the model with chunk-based pre-filling.

  * **Setting 3: w/ Chunk-based Pre-filling + Bi-level Decoding**

      * **Directory:** `./Video-XL-2/eval/lvu/w_chunk_bilevel/scripts`
      * This setting combines chunk-based pre-filling with bi-level decoding for maximum performance.

-----

### **2.2 Need in a Haystack (NIAH)**

*TODO*

-----

### **2.3 Temporal Grounding**

*TODO*
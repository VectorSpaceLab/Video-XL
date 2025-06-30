import sys
sys.path.append('.')
from videoxl2.train.train import train
import wandb
import os
wandb.init(mode="disabled")
wandb.login(key='Your_Key')
# 设置超时时间为60秒
os.environ["WANDB__SERVICE_WAIT"] = "600"
if __name__ == "__main__":
    train()

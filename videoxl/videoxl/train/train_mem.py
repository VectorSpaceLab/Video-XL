from videoxl.train.train import train
import wandb
import os
import wandb
wandb.login(key='1547db62a05790f3f88c42c2de42d12c77add5b4')
# 设置超时时间为60秒
os.environ["WANDB__SERVICE_WAIT"] = "600"
if __name__ == "__main__":
    train()



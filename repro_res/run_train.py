import argparse

import torch
import torch.nn as nn

from repro_res.constants import RANDOM_SEED
from repro_res.models import ResidualModel,SequentialModel
from repro_res.train import train_model,lr_schedule_default,lr_schedule_110

TOTAL_ITER = 64_000

def get_model_name(n : int, res : bool) -> str:
    # 6n+2 weighted layers, e.g. n=3 -> resnet-20 / plain-20
    return f"{'resnet' if res else 'plain'}-{6*n+2}"

def main():
    parser = argparse.ArgumentParser(description="Train a CIFAR-10 plain/residual net with 6n+2 layers")
    parser.add_argument("n",type=int,help="blocks per stage (3,5,7,9 -> 20,32,44,56 layers; 18 -> 110)")
    parser.add_argument("--res",action="store_true",help="residual model (default: plain)")
    args = parser.parse_args()

    # Seed before building the model so the initial weights are reproducible
    torch.manual_seed(RANDOM_SEED)
    model = ResidualModel(args.n) if args.res else SequentialModel(args.n)

    # The paper warms up at lr 0.01 for 110+ layers
    lr_fn = lr_schedule_110 if args.n >= 18 else lr_schedule_default

    train_model(model,get_model_name(args.n,args.res),nn.CrossEntropyLoss(),total_iter=TOTAL_ITER,lr_fn=lr_fn)

if __name__ == "__main__":
    main()

from functools import partial
from typing import Literal

import torch,torchvision

from torchvision.transforms import v2
from torch.utils.data import Dataset,DataLoader

from repro_res.constants import DATA_DIR,RESULTS_DIR,BATCH_SIZE

PER_PIXEL_AVERAGE_PATH = RESULTS_DIR /"cifar10_per_pixel_average.pt"

def compute_per_pixel_average():
    dataset = torchvision.datasets.CIFAR10(DATA_DIR,train=True,download=True)

    # dataset.data is uint8 (N,H,W,C); average over N and permute to (C,H,W)
    data = torch.from_numpy(dataset.data).to(torch.float64)
    per_pixel_average = data.mean(dim=0).permute(2,0,1).to(torch.float32).contiguous()

    PER_PIXEL_AVERAGE_PATH.parent.mkdir(parents=True,exist_ok=True)
    torch.save(per_pixel_average,PER_PIXEL_AVERAGE_PATH)
    return per_pixel_average

def get_per_pixel_average():
    # Check File existence, if exists read per pixel average from file, else compute.
    if PER_PIXEL_AVERAGE_PATH.exists():
        return torch.load(PER_PIXEL_AVERAGE_PATH)
    return compute_per_pixel_average()

def subtract_per_pixel_average(x,per_pixel_average : torch.Tensor):
    return x - per_pixel_average

def get_dataset(split : Literal["train","test"]):
    subtract_mean = partial(subtract_per_pixel_average,per_pixel_average=get_per_pixel_average())

    train_transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32,scale=False),
        subtract_mean,
        v2.Pad(4),
        v2.RandomHorizontalFlip(),
        v2.RandomCrop(32),
    ])

    test_transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32,scale=False),
        subtract_mean,
    ])
    
    if split == "train":
        return torchvision.datasets.CIFAR10(DATA_DIR,train=True,download=True,transform=train_transform)
    else:
        return torchvision.datasets.CIFAR10(DATA_DIR,train=False,download=True,transform=test_transform)

def get_dataloader(split : Literal["train","test"], batch_size : int = BATCH_SIZE):
    dataset = get_dataset(split)
    
    if split == "train":
        return DataLoader(dataset,batch_size=batch_size,shuffle=True)
    else:
        return DataLoader(dataset,batch_size=batch_size)
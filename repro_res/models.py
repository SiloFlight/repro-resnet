from typing import Any

import torch.nn as nn
import torch.nn.functional as F

class SequentialModel(nn.Module):
    """Accepts (B,C,H,W)"""
    def _create_layer_seq(self,*, layers :int, in_channels : int, out_channels : int, ) -> nn.Sequential:
        L = []
        for _ in range(layers):
            L += [
                nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,padding=1,bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            ]
        
        return nn.Sequential(*L)
    
    
    def __init__(self, n : int) -> None:
        if n <= 0:
            raise ValueError("Sequential Model created with non positive n")
        
        super().__init__()
        
        # Initial 3x3 32x32,16 filt
        self.l1 = nn.Sequential(
             nn.Conv2d(in_channels=3,out_channels=16,kernel_size=3,padding=1,bias=False),
             nn.BatchNorm2d(16),
             nn.ReLU()
        )
        
        # 2n 3x3 32x32, 16 filter
        self.block_1 = self._create_layer_seq(layers=2*n,in_channels=16,out_channels=16)
        
        # 3x3 stride 2, 32 filter
        self.down_sample_1 = nn.Sequential(
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        # 2n-1, 3x3 32 filter 16x16
        self.block_2 = self._create_layer_seq(layers=2*n-1,in_channels=32,out_channels=32)
        # 3x3 stride 2, 64 Filter
        self.down_sample_2 = nn.Sequential(
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        # 2n-1 3x3, 64 filters
        self.block_3 = self._create_layer_seq(layers=2*n-1,in_channels=64,out_channels=64)
        
        # GAP
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        # (B,64,1,1) -> (B,64)
        self.flatten = nn.Flatten()
        # FC
        self.fc = nn.Linear(in_features=64,out_features=10)

        self.__init_weights()

    def __init_weights(self) -> None:
        # He et al. Conv Initialization
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode="fan_in",nonlinearity="relu")

    def forward(self, x):
        x = self.l1(x)
        x = self.block_1(x)
        x = self.down_sample_1(x)
        x = self.block_2(x)
        x = self.down_sample_2(x)
        x = self.block_3(x)
        x = self.pool(x)
        x = self.flatten(x)
        return self.fc(x)
    
class ResidualConvBlock(nn.Module):
    def _downsample(self, x):
        if self.in_channels == self.out_channels and self.stride == 1:
            return x
        else:
            # Identity shortcut
            x = x[:,:,::self.stride,::self.stride]
            return F.pad(x,(0,0,0,0,0,self.out_channels - self.in_channels))
    
    def __init__(self, *, in_channels : int, out_channels : int, stride : int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,padding=1,stride=stride,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm2d(out_channels),
        )
        
        self.relu = nn.ReLU()

    def forward(self,x):
        out = self.conv1(x)
        out = self.conv2(out)
        
        out += self._downsample(x)
        
        return self.relu(out)

class ResidualModel(nn.Module):
    def _create_layer_seq(self, n : int, channels : int):
        L = []
        
        for i in range(n):
            L += [
                ResidualConvBlock(in_channels=channels,out_channels=channels,stride=1)
                ]
        
        return nn.Sequential(*L)
    
    def __init__(self, n : int) -> None:
        super().__init__()
        
        if n <= 0:
            raise ValueError("Sequential Model created with non positive n")
        
        self.l1 = nn.Sequential(
             nn.Conv2d(in_channels=3,out_channels=16,kernel_size=3,padding=1,bias=False),
             nn.BatchNorm2d(16),
             nn.ReLU()
        )
        
        self.block_1 = self._create_layer_seq(n, channels=16)
        
        self.downsample_1 = ResidualConvBlock(in_channels=16,out_channels=32,stride=2)
        
        self.block_2 = self._create_layer_seq(n-1,channels=32)
        
        self.downsample_2 = ResidualConvBlock(in_channels=32,out_channels=64,stride=2)
        
        self.block_3 = self._create_layer_seq(n-1,channels=64)
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        # (B,64,1,1) -> (B,64)
        self.flatten = nn.Flatten()
        # FC
        self.fc = nn.Linear(in_features=64,out_features=10)

        self.__init_weights()

    def __init_weights(self) -> None:
        # He et al. Conv Initialization
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode="fan_in",nonlinearity="relu")

    def forward(self, x):
        x = self.l1(x)
        x = self.block_1(x)
        x = self.downsample_1(x)
        x = self.block_2(x)
        x = self.downsample_2(x)
        x = self.block_3(x)
        x = self.pool(x)
        x = self.flatten(x)
        return self.fc(x)

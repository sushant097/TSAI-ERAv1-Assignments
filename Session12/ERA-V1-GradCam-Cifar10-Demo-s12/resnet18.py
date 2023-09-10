from pytorch_lightning import LightningModule, Trainer, seed_everything
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
from torchvision import transforms
from pytorch_lightning import LightningModule, Trainer, seed_everything


class LitResnet(LightningModule):
    def __init__(self, lr=0.01, drop= 0.2, norm='BN',groupsize=1):
        super().__init__()

        self.save_hyperparameters()
        self.num_classes =10
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            self.self_norm(norm,64,groupsize),
            nn.Dropout(drop),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1, bias=False),
            nn.MaxPool2d(2,2),
            nn.ReLU(),
            self.self_norm(norm,128,groupsize),
            nn.Dropout(drop))
        self.res1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            self.self_norm(norm,128,groupsize),
            nn.Dropout(drop),
            nn.Conv2d(in_channels = 128, out_channels=128, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            self.self_norm(norm,128,groupsize),
            nn.Dropout(drop)
             )

        # CONVOLUTION BLOCK 2
      	# Layer 2
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            self.self_norm(norm,256,groupsize),
            nn.Dropout(drop),
	    nn.MaxPool2d(2,2),
            nn.ReLU(),
            self.self_norm(norm,256,groupsize),
            nn.Dropout(drop)

             )

        # CONVOLUTION BLOCK 3
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=1, bias=False),
	    nn.MaxPool2d(2,2),
            nn.ReLU(),
            self.self_norm(norm,512,groupsize),
            nn.Dropout(drop)
            )
        self.res2 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            self.self_norm(norm,512,groupsize),
            nn.Dropout(drop),
            nn.Conv2d(in_channels = 512, out_channels=512, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            self.self_norm(norm,512,groupsize),
            nn.Dropout(drop))

        # CONVOLUTION BLOCK 4
        self.convblock4 = nn.Sequential(
             nn.MaxPool2d(kernel_size=4),
             nn.Conv2d(in_channels=512, out_channels=10, kernel_size=(1, 1), padding=0, bias=False))
        self.model = nn.Sequential(self.convblock1 ,self.convblock2, self.convblock3, self.convblock4)

    def self_norm(self, norm, channels,groupsize=1):
        if norm == 'BN':
            return nn.BatchNorm2d(channels)
        elif norm == 'LN':
            return nn.GroupNorm(1,channels) #(equivalent with LayerNorm)
        elif norm == 'GN':
            return nn.GroupNorm(groupsize,channels) #groups=2

    def forward(self, x):

        x = self.convblock1(x)
        x = x + self.res1 (x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = x + self.res2 (x)
        x = self.convblock4(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
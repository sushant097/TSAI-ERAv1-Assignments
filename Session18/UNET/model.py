import torch
import torch.nn as nn


class ContractingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pool="maxpool"):
        super(ContractingBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

        if pool == "maxpool":
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        elif pool == "strided_conv":
            self.pool = nn.Conv2d(out_channels, out_channels, kernel_size=2, stride=2, padding=0)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        skip = x  # store the output for the skip connection
        
        x = self.pool(x)        
        
        return x, skip

class ExpandingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mode="transpose"):
        super(ExpandingBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        
        
        if mode == "transpose":
            self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

        elif mode == "upsample":
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(in_channels, out_channels, kernel_size=1,stride=1)
            )

    def forward(self, x, skip):
        
        x = self.upsample(x)
        x = torch.cat((x, skip), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)        

        return x

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, pool="maxpool", mode="transpose"):
        super(UNet, self).__init__()
        
        self.contract1 = ContractingBlock(in_channels, 64, pool)
        self.contract2 = ContractingBlock(64, 128, pool)
        self.contract3 = ContractingBlock(128, 256, pool)
        self.contract4 = ContractingBlock(256, 512, pool)
     
        self.expand1 = ExpandingBlock(512, 256, mode)
        self.expand2 = ExpandingBlock(256, 128, mode)
        self.expand3 = ExpandingBlock(128, 64, mode)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)        
        
    def forward(self, x):
        # Contracting path
        x, skip1 = self.contract1(x)
        x, skip2 = self.contract2(x)
        x, skip3 = self.contract3(x)
        _, x = self.contract4(x)
        
        # Expanding path
        x = self.expand1(x, skip3)
        x = self.expand2(x, skip2)
        x = self.expand3(x, skip1)

        x = self.final_conv(x)
        return x
     

class DiceLoss(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        # flatten predictions and targets
        pred = pred.view(-1)
        target = target.view(-1)

        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()

        dice = (2.0 * intersection + self.eps) / (union + self.eps)

        return 1 - dice
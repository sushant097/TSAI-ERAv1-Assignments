import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)


train_transforms = transforms.Compose(
        [
            transforms.Resize((128, 128),
            interpolation=transforms.InterpolationMode.NEAREST_EXACT),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ]
)

target_transforms = transforms.Compose(
    [
        transforms.PILToTensor(),                                        
        transforms.Resize((128, 128), interpolation=transforms.InterpolationMode.NEAREST_EXACT),
        transforms.Lambda(lambda x: (x-1).squeeze().type(torch.LongTensor))
    ]
)


class Oxford_Pet_dataset(datasets.OxfordIIITPet):
    def __init__(self, root="./data", split="trainval", target_types="segmentation", train_transform=None, target_transform=None, download=True):
        super().__init__(root=root, split=split, target_types=target_types, download=download, transform=train_transform)
        self.train_transform = train_transform
        self.target_transform = target_transform  

    def __getitem__(self, index):
        images, labels = self._images[index], self._segs[index]
        image = Image.open(images).convert("RGB")
        label = Image.open(labels)
    
        image = self.train_transform(image)
        label = self.target_transform(label)
        one_hot_label = torch.nn.functional.one_hot(label, 3).transpose(0, 2).squeeze(-1).transpose(1, 2).squeeze(-1)
        return image, label, one_hot_label    
    

class OxfordPetDataModule(LightningDataModule):
    def __init__(self, data_dir, batch_size, num_workers):
        super(OxfordPetDataModule, self).__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
    
    def prepare_data(self):
        pass

    def setup(self, stage="train"):
        self.train_ds = Oxford_Pet_dataset(
            root="./data", split="trainval", target_types="segmentation", train_transform=train_transforms, 
            target_transform=target_transforms, download=True
        )

        self.test_ds = Oxford_Pet_dataset(
            root="./data", split="test", target_types="segmentation", train_transform=train_transforms, 
            target_transform=target_transforms, download=True
        )

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    
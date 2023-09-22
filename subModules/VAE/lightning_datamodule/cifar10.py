from torchvision import datasets, transforms
import pytorch_lightning as pl
import torch

transform = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ]
)

class CIFAR10Dataset(datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(CIFAR10Dataset, self).__init__(root, train, transform, target_transform, download)

    def __getitem__(self, index):
        img, target = super(CIFAR10Dataset, self).__getitem__(index)              
        return img, target
    
class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, num_workers):
        super().__init__()       
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        pass

    def setup(self,stage):
        self.train_ds = CIFAR10Dataset(
            root=self.data_dir, 
            train=True, 
            transform=transform, 
            download=True
        )
        
        self.test_ds = CIFAR10Dataset(
            root=self.data_dir, 
            train=False, 
            transform=transform
        )
                 
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
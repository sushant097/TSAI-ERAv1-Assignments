from torchvision import datasets, transforms
import pytorch_lightning as pl
import torch

transform = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]
)

class MNISTtoRGB(datasets.MNIST):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(MNISTtoRGB, self).__init__(root, train, transform, target_transform, download)

    def __getitem__(self, index):
        img, target = super(MNISTtoRGB, self).__getitem__(index)
        img = torch.cat((img, img, img), 0)  # Convert single channel to 3 channels (R, G, B)
    
        return img, target
    
class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, num_workers):
        super().__init__()       
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        pass 

    def setup(self,stage):
        self.train_ds = MNISTtoRGB(
            root=self.data_dir, 
            train=True, 
            transform=transform, 
            download=True
        )

        self.test_ds = MNISTtoRGB(
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
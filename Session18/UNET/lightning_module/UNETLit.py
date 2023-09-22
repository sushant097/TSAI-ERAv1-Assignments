import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
import matplotlib.pyplot as plt

from model import UNet, DiceLoss


device = 'cuda' if torch.cuda.is_available() else 'cpu'

class UNETLitModel(LightningModule):

    def __init__(self, in_channels, out_channels, pool, mode, criterion="bce"):
        super(UNETLitModel, self).__init__()

        self.net = UNet(in_channels, out_channels, pool, mode)

        self.train_loss = []
        self.train_loss_plot = []

        if criterion == "bce":
            self.loss_fn = nn.BCEWithLogitsLoss()
        elif criterion == "dice":
            self.loss_fn = DiceLoss()
        elif criterion == "cross-entropy":
            self.loss_fn = nn.CrossEntropyLoss()        
    
    def forward(self, x):
        return self.net(x)
    

    def training_step(self, batch, batch_idx):
        x, y, one_hot_labels = batch
        one_hot_labels = one_hot_labels.type(torch.FloatTensor)

        x, y, one_hot_labels = x.to(device), y.to(device), one_hot_labels.to(device)

        logits = self.forward(x)
        loss = self.loss_fn(logits, one_hot_labels)
        
        self.train_loss.append(loss.item())
        mean_loss = sum(self.train_loss) / len(self.train_loss)
        self.log("Train_loss", mean_loss, prog_bar=True)

        return loss
    
    
    def on_train_epoch_end(self):
        mean_loss = sum(self.train_loss) / len(self.train_loss)
        self.train_loss_plot.append(mean_loss)

        print(f"Epoch: {self.current_epoch+1} | Loss: ",round(mean_loss, 5))
        
        self.train_loss = []


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3, eps=1e-9)

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=1e-3,
            total_steps=self.trainer.estimated_stepping_batches,
            epochs=self.trainer.max_epochs,
            pct_start=0.3,
            div_factor=10,
            three_phase=True,
            final_div_factor=10,
            anneal_strategy="linear",
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                'interval': 'step',  
                'frequency': 1
            },
        }
    

    def plot_train_loss(self):
        epochs = list(range(1, len(self.train_loss_plot) + 1))

        plt.figure(figsize=(10, 5))
        plt.plot(epochs, self.train_loss_plot, label='Train Loss', color='red')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Epochs')
        plt.legend()
        plt.grid(True)        
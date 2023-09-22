import os
import pytorch_lightning as pl
from torchsummary import summary

from lightning_module.UNETLit import UNETLitModel
from lightning_module.datamodule import OxfordPetDataModule
from model import UNet
from utils import *
from config import *

dm = OxfordPetDataModule(
    data_dir=DATA_DIR,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS
)

dm.setup()
train_loader = dm.train_dataloader()
test_loader = dm.test_dataloader()

def run(model):

    trainer = pl.Trainer(
        precision="16-mixed",
        max_epochs=NUM_EPOCHS,
        accelerator="gpu"
    )

    trainer.fit(model, dm)
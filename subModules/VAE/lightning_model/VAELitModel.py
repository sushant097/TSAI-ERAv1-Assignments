import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn import functional as F
from pl_bolts.models.autoencoders.components import (
    resnet18_decoder,
    resnet18_encoder,
)
from torch.optim.lr_scheduler import OneCycleLR

class VAELitModel(pl.LightningModule):
    def __init__(self, enc_out_dim=512, latent_dim=256, input_height=32, num_embed=10):
        super().__init__()

        self.save_hyperparameters()
        self.train_loss = []

        # encoder, decoder
        self.encoder = resnet18_encoder(False, False)
        self.decoder = resnet18_decoder(
            latent_dim=latent_dim,
            input_height=input_height,
            first_conv=False,
            maxpool1=False
        )

        # distribution parameters
        self.fc_mu = nn.Linear(enc_out_dim, latent_dim)
        self.fc_var = nn.Linear(enc_out_dim, latent_dim)

        # for the gaussian likelihood
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))

        #label embedding
        self.label_embedding  = nn.Embedding(num_embed, embedding_dim=enc_out_dim)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)      
        
        self.scheduler = OneCycleLR(
            optimizer, 
            max_lr=1e-3,
            steps_per_epoch=len(self.trainer.datamodule.train_dataloader()),
            epochs=self.trainer.max_epochs, 
            pct_start=5/self.trainer.max_epochs,
            three_phase=True,
            div_factor=10,
            final_div_factor=10,
            anneal_strategy='linear',
        )
        
        return {
          "optimizer": optimizer, 
          "lr_scheduler": {
            "scheduler": self.scheduler ,
            "interval": "step",
          }
        }

    def gaussian_likelihood(self, mean, logscale, sample):
        scale = torch.exp(logscale)
        dist = torch.distributions.Normal(mean, scale)
        log_pxz = dist.log_prob(sample)
        return log_pxz.sum(dim=(1, 2, 3))

    def kl_divergence(self, z, mu, std):
        # --------------------------
        # Monte carlo KL divergence
        # --------------------------
        # 1. define the first two probabilities (in this case Normal for both)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)

        # 2. get the probabilities from the equation
        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)

        # kl
        kl = (log_qzx - log_pz)
        kl = kl.sum(-1)
        return kl

    def forward(self, x):
        x, label = x
        x_encoded = self.encoder(x)
        x_encoded_embedded = x_encoded * self.label_embedding(label)
        mu, log_var = self.fc_mu(x_encoded_embedded), self.fc_var(x_encoded_embedded)

        # sample z from q
        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()

        # decoded
        x_hat = self.decoder(z)
        return x_hat

    def training_step(self, batch, batch_idx):
        x, label = batch

        # encode x to get the mu and variance parameters
        x_encoded = self.encoder(x)
        x_encoded_embedded = x_encoded * self.label_embedding(label)

        mu, log_var = self.fc_mu(x_encoded_embedded), self.fc_var(x_encoded_embedded)

        # sample z from q
        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()

        # decoded
        x_hat = self.decoder(z)

        # reconstruction loss
        recon_loss = self.gaussian_likelihood(x_hat, self.log_scale, x)

        # kl
        kl = self.kl_divergence(z, mu, std)

        # elbo
        elbo = (kl - recon_loss)
        elbo = elbo.mean()

        self.log_dict(
            {
                "elbo": elbo,
                "kl": kl.mean(),
                "recon_loss": recon_loss.mean(),
                "reconstruction": recon_loss.mean(),
            }
        )

        self.train_loss.append(elbo)
        return elbo
    
    def on_train_epoch_end(self):
        mean_loss = sum(self.train_loss) / len(self.train_loss)
        print(f"Epoch: {self.current_epoch+1} | Loss: {mean_loss}")
        self.train_loss = []

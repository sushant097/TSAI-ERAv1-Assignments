import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import LambdaLR
import torchmetrics
from torchmetrics import BLEUScore, CharErrorRate, WordErrorRate

from model import build_transformer
from utils import causal_mask

from config import get_config

config_ = get_config()


class BilangLightning(LightningModule):
    def __init__(self, learning_rate, tokenizer_src, tokenizer_tgt, config=config_):

        super().__init__()

        self.learning_rate = learning_rate
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.config = config

        self.seq_len = config["seq_len"]

        self.src_vocab_size = tokenizer_src.get_vocab_size()
        self.tgt_vocab_size = tokenizer_tgt.get_vocab_size()

        # self.src_vocab_size = 15698
        # self.tgt_vocab_size = 22463

        self.last_batch = None

        self.net = build_transformer(
            self.src_vocab_size,
            self.tgt_vocab_size,
            self.seq_len,
            self.seq_len,
            d_model=config["d_model"],
        )

        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=self.tokenizer_src.token_to_id("[PAD]"), label_smoothing=0.1
        )

        self.epoch_condition = 0

        self.train_loss = []
        self.val_loss = []

        self.source_texts = []
        self.expected = []
        self.predicted = []

        self.char_error_rate = CharErrorRate()
        self.word_error_rate = WordErrorRate()
        self.bleu_score = BLEUScore()

    def forward(self, encoder_input, decoder_input, encoder_mask, decoder_mask):
        encoder_output = self.net.encode(
            encoder_input, encoder_mask
        )  # (B, seq_len, seq_len)
        decoder_output = self.net.decode(
            encoder_output, encoder_mask, decoder_input, decoder_mask
        )
        proj_output = self.net.project(decoder_output)  # (B, seq_len, vocab_size)

        return proj_output

    def training_step(self, batch, batch_idx):
        encoder_input = batch["encoder_input"]
        decoder_input = batch["decoder_input"]
        encoder_mask = batch["encoder_mask"]
        decoder_mask = batch["decoder_mask"]
        proj_output = self.forward(
            encoder_input, decoder_input, encoder_mask, decoder_mask
        )

        label = batch["label"]  # (B, seq_len)
        loss = self.loss_fn(
            proj_output.view(-1, self.tokenizer_tgt.get_vocab_size()), label.view(-1)
        )

        # update and log metrics
        self.train_loss.append(loss)
        mean_loss = sum(self.train_loss) / len(self.train_loss)
        self.log("Train Loss", mean_loss, prog_bar=True)

        self.epoch_condition = 1

        return loss
    
    def on_train_epoch_end(self):
        mean_train_loss = sum(self.train_loss) / len(self.train_loss)
        print("Training Loss   : ", round(mean_train_loss.item(), 5))
        print("----------------------------------------------------------------------")
        self.train_loss = []


    def validation_step(self, batch, batch_idx):
        """
        encoder_input = batch["encoder_input"]
        decoder_input = batch["decoder_input"]
        encoder_mask = batch["encoder_mask"]
        decoder_mask = batch["decoder_mask"]
        proj_output = self.forward(
            encoder_input, decoder_input, encoder_mask, decoder_mask
        )

        label = batch["label"]  # (B, seq_len)
        loss = self.loss_fn(
            proj_output.view(-1, self.tokenizer_tgt.get_vocab_size()), label.view(-1)
        )

        # update and log metrics
        self.val_loss.append(loss)
        mean_loss = sum(self.val_loss) / len(self.val_loss)
        self.log("Val Loss", mean_loss, prog_bar=True)
        """
        self.last_batch = batch

    def on_validation_epoch_end(self) -> None:
        if self.epoch_condition > 0:

            print(f"Epoch : {self.current_epoch}")

            # train_mean_loss = sum(self.train_loss) / len(self.train_loss)
            # print(f"Training Loss : {train_mean_loss:5f}")

            # val_mean_loss = sum(self.val_loss) / len(self.val_loss)
            # print(f"Validation Loss : {val_mean_loss:5f}")

            self.run_validation(self, self.last_batch)

            # self.train_loss = []
            # self.val_loss = []

            self.epoch_condition = 0

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config['lr'], eps=1e-9)

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.config["lr"],
            epochs=self.trainer.max_epochs,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.3,
            div_factor=10,
            final_div_factor=10,
            three_phase=True,
            anneal_strategy='linear',
            verbose=False
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                'interval': 'step',  
                'frequency': 1
            },
        }

    def greedy_decode(self, source, source_mask, max_len: int, device: str):
        sos_idx = self.tokenizer_tgt.token_to_id("[SOS]")
        eos_idx = self.tokenizer_tgt.token_to_id("[EOS]")

        # encoder output
        encoder_output = self.net.encode(source, source_mask)
        decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
        while True:
            if decoder_input.size(1) == max_len:
                break
            # build target mask
            decoder_mask = (
                causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
            )
            out = self.net.decode(
                encoder_output, source_mask, decoder_input, decoder_mask
            )

            # get next token
            prob = self.net.project(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            decoder_input = torch.cat(
                [
                    decoder_input,
                    torch.empty(1, 1)
                    .type_as(source)
                    .fill_(next_word.item())
                    .to(device),
                ],
                dim=1,
            )
            if next_word == eos_idx:
                break

        return decoder_input.squeeze(0)

    def run_validation(self, model, data):
        model.eval()

        src = data["encoder_input"]
        src_mask = data["encoder_mask"]

        model_out = self.greedy_decode(
            src, src_mask, max_len=self.seq_len, device=self.device
        )

        source_text = data["src_text"]
        target_text = data["tgt_text"]
        model_out_text = self.tokenizer_tgt.decode(model_out.detach().cpu().numpy())

        expected = target_text
        predicted = model_out_text

        metric = CharErrorRate()
        cer = metric(predicted, expected)

        metric = WordErrorRate()
        wer = metric(predicted, expected)

        metric = BLEUScore()
        bleu = metric(predicted, expected)

        print("----------------------------------------------------------------------")
        print(f"SOURCE    => {source_text}")
        print(f"Ground Truth  => {expected}")
        print(f"PREDICTED => {predicted}")
        print("----------------------------------------------------------------------")
        print(f"Validation CER  => {cer}")
        print(f"Validation WER  => {wer}")
        print(f"Validation BLEU => {bleu}")
        print("----------------------------------------------------------------------")

        self.log("Validation - CER", cer, prog_bar=True)
        self.log("Validation - WER", wer, prog_bar=True)
        self.log("Validation - BLEU", bleu, prog_bar=True)

        model.train()

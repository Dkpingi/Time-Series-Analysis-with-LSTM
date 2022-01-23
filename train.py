import torch
import pytorch_lightning as pl
from lstm import LSTM, MultiScaleLSTM
from dataset import CoinPriceDataModule, CoinPriceDataset
from data import get_all_historic_data
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import ModelCheckpoint

seq_lens = [5, 30, 60, 90, 120]
batch_size = 100000 #all
lr = 1e-4
weight_decay = 1e-4
max_epochs = 10000000 #until stop
hidden_dim = 5
in_channels = [3]
n_in_channels = 1
n_layers = 1
out_channels = [3]
n_out_channels = 1
hours = 24

model = MultiScaleLSTM(in_channels = in_channels, seq_lens = seq_lens, out_channels = out_channels, hidden_dim = hidden_dim, 
             n_layers=n_layers,  lr=lr, wd = weight_decay, n_in_channels = n_in_channels, n_out_channels = n_out_channels,
             batch_size = batch_size, hours = hours)

checkpoint_callback = ModelCheckpoint(monitor="val_loss")

trainer = pl.Trainer(
    progress_bar_refresh_rate=10,
    max_epochs=max_epochs,
    gpus=1,
    callbacks=[checkpoint_callback],
    logger=pl.loggers.TensorBoardLogger("lightning_logs/", name="lstm"),
)

trainer.fit(model)

dataset = CoinPriceDataset(in_channels, seq_len, 'val')
rates = dataset.data
rates_last = torch.tensor(rates[-seq_len:, :in_channels]).unsqueeze(0).float()
current_price = rates_last[0, -1, 0]

print(model.predict(rates_last))

import torch
import pytorch_lightning as pl
from lstm import LSTM
from dataset import CoinPriceDataModule, CoinPriceDataset
from data import get_all_historic_data
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np

seq_len = 90
batch_size = 100000 #all
lr = 1e-4
weight_decay = 1e-4
max_epochs = 400000
hidden_dim = 10
in_channels = 2
n_layers = 2
out_channels = 2

seq_len_ = [5, 10, 30, 60, 90, 120, 300]
lr_ = [0.1, 0.5, 0.01, 0.05, 0.001, 0.005, 0.0001, 0.0005]
weight_decay_ = [1.0, 0.1, 0.5, 0.01, 0.05, 0.001, 0.005, 0.0001, 0.0005]
hidden_dim_ = [1, 2, 3, 5, 10, 20]
n_layers_ = [1, 2, 3, 5, 10]
in_channels_ = list(range(84))
out_channels_ = [3]
hours_ = [24]


#fixing some parameters after 300 runs, as they are clearly better than the rest:
lr_ = [1e-4]
seq_len_ = [90]
n_layers_ = [1, 2, 3]
hidden_dim_ = [1, 2, 3]
weight_decay_ = list(np.arange(0.0, 1.0, 0.01))


opt_dict = {'seq_len' : seq_len_, 'lr' : lr_, 'wd' : weight_decay_, 'hidden_dim' : hidden_dim_, 'n_layers' : n_layers_, 
           'in_channels' : in_channels_, 'out_channels' : out_channels_, 'hours' : hours_}

def randomize_params(inputs):
    params = {}
    for key, value in inputs.items():
        params[key] = np.random.choice(inputs[key])
    params['n_out_channels'] = 1
    params['n_in_channels'] = np.random.randint(1, len(inputs['in_channels']))
    params['batch_size'] = 100000

    params['in_channels'] = np.random.choice(inputs['in_channels'], size = params['n_in_channels'], replace = False)
    if 3 not in params['in_channels']:
        params['in_channels'] = np.append(params['in_channels'], 3)
        params['n_in_channels'] += 1
    params['out_channels'] = np.array(inputs['out_channels'])

    print(params)
    return params


for i in range(0, 100):
    try: #in case of bad param
        params = randomize_params(opt_dict)
        model = LSTM(**params)       

        checkpoint_callback = ModelCheckpoint(monitor="val/rel_err")
        early_stop_callback = EarlyStopping(monitor="val/rel_err", min_delta=0.00001, patience=500, verbose=False, mode="min")

        trainer = pl.Trainer(
            progress_bar_refresh_rate=10,
            max_epochs=max_epochs,
            gpus=1,
            callbacks=[checkpoint_callback, early_stop_callback],
            logger=pl.loggers.TensorBoardLogger("lightning_logs/", name="hparamsearch"),
        )

        trainer.fit(model)
        
        in_channels, out_channels, seq_len, hours = params['in_channels'], params['out_channels'], params['seq_len'], params['hours']
        
        dataset = CoinPriceDataset(in_channels, out_channels, seq_len, 'val', hours)
        batch = dataset[dataset.__len__() - 1]
        
        print(model.predict(batch))
    except Exception as e:
        print(e)
        continue

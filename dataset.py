from torch.utils.data import Dataset, DataLoader
from datetime import datetime
import pytorch_lightning as pl
import pickle as pkl
import numpy as np
import pandas as pd
from data import get_all_historic_data
from ta import add_all_ta_features
from ta.utils import dropna

class CoinPriceDataset(Dataset):
    def __init__(self, in_channels, out_channels, length, split, hours):
        super().__init__()
        get_all_historic_data(hours = hours)
        self.split = split
        self.length = length
        self.in_channels = in_channels
        self.out_channels = out_channels
        with open('rates.pkl', 'rb') as f:
            self.data = pkl.load(f).astype('float32')
        
        # Load datas
        df = pd.DataFrame(self.data[:, 1:], columns=['low', 'high', 'open', 'close', 'volume'])

        # Clean NaN values
        df = dropna(df)

        # Add ta features filling NaN values
        df = add_all_ta_features(
        df, open="open", high="high", low="low", close="close", volume="volume", fillna=True)
        
        dropcolumns = []
        for column in df:
            if (df[column] != 0).sum() < 0.9*len(df[column]):
                dropcolumns.append(column)
        df = df.drop(columns = dropcolumns)
        ta_array = df.to_numpy()

        timearray = np.zeros((self.data.shape[0], 3))
        for i, ts in enumerate(self.data[:, 0]):
            dt = datetime.utcfromtimestamp(ts)
            weekday = dt.isoweekday()/7.0
            month = dt.month/12.0
            day = dt.day/31.0
            timearray[i, :] = day, weekday, month

        #self.data = self.data[:, 1:]
        
        #self.data = np.delete(self.data, 3, axis=1)

        self.data = np.concatenate([ta_array, timearray], axis = 1).astype('float32')

        targets = self.data[self.length:, :]
        l = targets.shape[0]

        self.boundary = int(np.floor(0.8*l))
        if self.split == 'val':
            self.offset = self.boundary
        else:
            self.offset = 0
        
        self.targets = {}
        self.targets['train'] = targets[:self.boundary]
        self.targets['val'] = targets[self.boundary:]
                 
    def __len__(self):
        return len(self.targets[self.split])
    

    def __getitem__(self, idx):
        inputs = np.copy(self.data[self.offset + idx: self.length + self.offset + idx])
        self.scale = np.max(np.abs(inputs), axis = 0, keepdims = True)
        np.where(self.scale != 0.0, self.scale, 1.0)

        inputs /= self.scale
        inputs = inputs[:, self.in_channels]
        
        self.y_scale = self.scale[:, tuple(self.out_channels)]
        targets = np.copy(self.targets[self.split][idx, tuple(self.out_channels)])[None, :]
        targets /= self.y_scale
        return {'inputs' : inputs, 'targets': targets, 'y_scale' : self.y_scale}


class CoinPriceDataModule(pl.LightningDataModule):
    def __init__(self, in_channels = 1, out_channels = 1, seq_len = 90, batch_size = 32, hours = 24):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.hours = hours

    def prepare_data(self):
        pass

    def setup(self, stage = None):
        self.coin_train = CoinPriceDataset(self.in_channels, self.out_channels, self.seq_len, 'train', self.hours)
        self.coin_val = CoinPriceDataset(self.in_channels, self.out_channels, self.seq_len, 'val', self.hours)

    def train_dataloader(self):
        return DataLoader(self.coin_train, batch_size=self.batch_size, shuffle = True)

    def val_dataloader(self):
        return DataLoader(self.coin_val, batch_size=self.batch_size, shuffle = False)

if __name__ == '__main__':
    pass

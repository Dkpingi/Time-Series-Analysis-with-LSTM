

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import cbpro
import datetime
import numpy as np
import pickle as pkl
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from pytorch_lightning.metrics.functional import accuracy


client = cbpro.PublicClient()
rates = client.get_product_historic_rates('BTC-EUR', granularity=6*60*60, start='2019-09-15T00:00', end='2019-12-15T00:00')

def get_all_historic_data(pair = 'BTC-EUR', hours=6): #60*60*24
    
    granularity = int(60*60*hours)
    client = cbpro.PublicClient()
    time = client.get_time()['iso'].split('.')[0]
    enddate = datetime.datetime.fromisoformat(time)

    interval = 200*granularity
    startdate = enddate - datetime.timedelta(seconds = interval)
    rates = []
    print(interval)
    # Format: [ time, low, high, open, close, volume ]
    rate = client.get_product_historic_rates(pair, granularity=granularity, start=startdate, end=enddate)
    i = 0
    while len(rate) > 1:
        enddate = enddate - datetime.timedelta(seconds = interval)
        startdate = startdate - datetime.timedelta(seconds = interval)
        
        rates.insert(0, list(reversed(rate)))
        rate = client.get_product_historic_rates(pair, granularity=granularity, start=startdate, end=enddate)
    
    rates = [np.array(rate) for rate in rates]
    rates = np.concatenate(rates, axis = 0)
    with open('rates.pkl', 'wb') as f:
        pkl.dump(rates, f)
    return rates


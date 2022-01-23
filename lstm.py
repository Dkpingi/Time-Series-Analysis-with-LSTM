import torch
import torch.nn as nn
import pytorch_lightning as pl
from dataset import CoinPriceDataModule

class LSTM(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        kwargs = kwargs.pop('kwargs', kwargs)
        print(kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        
        self.c0, self.h0 = (torch.randn((self.n_layers, 1, self.hidden_dim)), torch.randn((self.n_layers, 1, self.hidden_dim))) 
        self.c0 = torch.nn.Parameter(self.c0, requires_grad = True)
        self.h0 = torch.nn.Parameter(self.h0, requires_grad = True)

        self.lstm = torch.nn.LSTM(self.n_in_channels, self.hidden_dim, self.n_layers, batch_first=True)
        self.linear = torch.nn.Linear(self.seq_len*self.hidden_dim, self.n_out_channels)        
        self.loss = torch.nn.MSELoss()
        
        cp_data = CoinPriceDataModule(self.in_channels, self.out_channels, self.seq_len, self.batch_size, self.hours)
        self.datamodule = cp_data
        
        conf = {'in_channels' : kwargs.pop('in_channels'), 'out_channels' : kwargs.pop('out_channels')}
        self.save_hyperparameters(kwargs)
        self.save_hyperparameters(conf)

        
    def norm(self, x):
        self.norm_factor = torch.max(x, dim = 1, keepdims = True)[0]
        x[:, :, :4] = x[:, :, :4]/ self.norm_factor[:, :, :4]
        return x

    def forward(self, x):
        c0 = torch.cat([self.c0]*x.shape[0], dim = 1)
        h0 = torch.cat([self.h0]*x.shape[0], dim = 1)
        x, (h0, c0) = self.lstm(x, (c0, h0))
        x = x.flatten(1)
        out = self.linear(x)
        return out.reshape(x.shape[0], 1, self.n_out_channels)

    def rel_error(self, x, y):
        return torch.mean(torch.abs((x-y)/y))
    
    def rel_bias(self, x, y):
        return torch.mean((x - y)/y)

    def training_step(self, batch, batch_idx):
        x, y = batch['inputs'], batch['targets']
        ypred = self(x)
        loss = self.loss(ypred, y)
        rel = self.rel_error(ypred, y)
        bias = self.rel_bias(ypred, y)

        self.log("train_loss", loss)
        self.log("rel_err", rel)
        self.log("rel_bias", bias)
        return loss

    def evaluate(self, batch, stage=None):
        x, y  = batch['inputs'], batch['targets']
        ypred = self(x)
        loss = self.loss(ypred, y)
        rel = self.rel_error(ypred, y)
        bias = self.rel_bias(ypred, y)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}/rel_err", rel)
            self.log(f"{stage}/rel_bias", bias)
            self.log("hp_metric", rel)

    def validation_step(self, batch, batch_idx):
        #print(self.predict(batch))
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def predict(self, batch):
        x, y_scale = batch['inputs'], batch['y_scale']
        x = torch.tensor(x).to(self.device)
        y_scale = torch.tensor(y_scale).to(self.device)
        ypred = self(x.unsqueeze(0))
        ypred *= y_scale.unsqueeze(0)
        return ypred

    def configure_optimizers(self):
        lr = self.lr
        wd = self.wd
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=lr,
            weight_decay=wd,
        )
        #scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, cycle_momentum=False, base_lr=0.2*lr, max_lr=5*lr, step_size_up = 1000)
        return {"optimizer": optimizer}



class MultiScaleLSTM(LSTM):
    def __init__(self, **kwargs):
        pl.LightningModule.__init__(self)
        kwargs = kwargs.pop('kwargs', kwargs)
        print(kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        self.models = torch.nn.ModuleList([])
        for seq_len in self.seq_lens:
            lstm = torch.nn.LSTM(self.n_in_channels, self.hidden_dim, self.n_layers, batch_first=True)
            linear = torch.nn.Linear(seq_len*self.hidden_dim, self.n_out_channels)        
            self.models.append(nn.Sequential(*[lstm, linear]))

        print(self.models)

        self.loss = torch.nn.MSELoss()
        
        cp_data = CoinPriceDataModule(self.in_channels, self.out_channels, self.seq_lens[4], self.batch_size, self.hours)
        self.datamodule = cp_data
        
        conf = {'in_channels' : kwargs.pop('in_channels'), 'out_channels' : kwargs.pop('out_channels')}
        #self.save_hyperparameters(kwargs)
        #self.save_hyperparameters(conf)

    def partial_forward(self, model, x):
        x, (h0, c0) = model[0](x)
        x = x.flatten(1)
        out = model[1](x)
        return out.reshape(x.shape[0], 1, self.n_out_channels)

    def forward(self, x):
        out = torch.zeros(x.shape[0], 1, self.n_out_channels).to(x)
        for idx, model in enumerate(self.models):
            out += self.partial_forward(model, x[:, :self.seq_lens[idx], :])
        return out

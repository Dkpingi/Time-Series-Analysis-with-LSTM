from dataset import CoinPriceDataset
from lstm import LSTM
import torch
import yaml
import cbpro
import time
import numpy as np
ckpt_file_path = './best_model/best_model.ckpt'

with open("./best_model/hparams.yaml", "r") as stream:
    try:
        params = yaml.unsafe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

print(params)

model = LSTM.load_from_checkpoint(ckpt_file_path)#, hparams_file='./best_model/hparams.yaml', conf = conf)
#get current prices

hours = 6
state = 'crypto'
order_id = None
last_order = None
auth_client = cbpro.AuthenticatedClient('XXXXXX', 'XXXXXX')

auth_client.cancel_all()
while True:
    
    #get prediction from model
    model.datamodule.setup()
    dataset = model.datamodule.coin_val
    batch = dataset[dataset.__len__() - 1]
    model.eval()

    prediction = model.predict(batch).detach()
    current_price = batch['targets']*batch['y_scale']
    signed_diff = (prediction - current_price)/current_price

    print('Current Price:', current_price)
    print('Prediction:', prediction.detach())
    print('Percent_difference:', signed_diff.detach())

    mo_profit = signed_diff[2]
    print(mo_profit)

    current_funds = auth_client.get_accounts()
    for f in current_funds:
        if f['currency'] == 'EUR':
            fiat_funds = np.around(float(f['available']), 2)
        if f['currency'] == 'BTC':
            crypto_funds = float(f['available'])

    print(fiat_funds)
    print(crypto_funds)
    exit()
    #execute trade
    if state == 'fiat' and fiat_funds > 10.0:
        '''
        if margin > 0.015:
            price = prediction[0]
            size = fiat_funds/price
            last_order = auth_client.place_limit_order(product_id='BTC-EUR', 
                                                        side='buy',
                                                        price=price,
                                                        size=size)
            print(last_order)
            order_id = last_order['id']
            state = 'crypto'
        '''
        if mo_profit > 0.015:
            last_order = auth_client.place_market_order(product_id='BTC-EUR', 
                                                    side='buy',
                                                    funds=fiat_funds)
            print(last_order)
            order_id = last_order['id']
            state = 'crypto'

    elif state == 'crypto' and crypto_funds > 0.004:
        '''
        if margin > 0.015:
            price = prediction[1]
            last_order = auth_client.place_limit_order(product_id='BTC-EUR', 
                                                    side='sell',
                                                    price=price,
                                                    size=crypto_funds)
            print(last_order)
            order_id = last_order['id']
            state = 'fiat'
        '''
        if mo_profit < -0.015:
            last_order = auth_client.place_market_order(product_id='BTC-EUR', 
                                                    side='sell',
                                                    size=crypto_funds)
            print(last_order)
            order_id = last_order['id']
            state = 'fiat'


    if last_order is not None:
        settled = False
        while not settled:
            if order_id is not None:
                order = auth_client.get_order(order_id)
                settled = order['settled']
            if not settled:
                time.sleep(60*60*5)
            else:
                last_order = None
    else:
        #print('Pred_margin:', margin)
        print('Preg Avg:', mo_profit)
        print('Trade not profitable. Waiting 30 Minutes')
        time.sleep(60*60*30)

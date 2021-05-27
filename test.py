import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import pickle as pkl
from binance.client import Client
import config

from stonks.cryptobot import *
from stonks.binance import plot_roc, get_binance_predictions
from stonks.util import balance_class

coins_train = coins_cleaned_topthousand
stocktype = 'crypto'

n_predictor_days = 15
scope = 'daily'
leave_out_days = 180

# mode = 'long'
# minimum_rate = 1.11
# maximum_loss = 0.95

mode = 'short'
minimum_rate = 0.89
maximum_loss = 1.05

print('load')
fn = f'C:/Users/Lukas/Documents/projects/stonks/assets/data/top1000_cleaned_train_DFs.pkl'
with open(fn, 'rb') as f:
    coin_price_dict = pkl.load(f)

x_tr, y_tr, x_test, y_test, _ = create_data(coin_price_dict, n_predictor_days, verbose=False, minimum_rate=minimum_rate, maximum_loss=maximum_loss, leave_out_days=leave_out_days, mode=mode)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from ..util import (get_train_test_data, normalize, scale_prediction, centering_normalizer, 
    normalize_candle_volume_marketcap, norm_separately, balance_class, remove_nan_rows)
from ..trends import get_daily_trend, trends_to_coin
import numpy as np
from cryptocmd import CmcScraper

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Dropout

from sklearn.metrics import roc_curve, roc_auc_score
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score, precision_score, recall_score

import plotly.graph_objects as go
from copy import deepcopy
import logging
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf  
from tqdm import tqdm

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
loss = 'mean_squared_error'
# loss = 'mean_absolute_error'

# predictor_variables = ['open', 'close', 'low', 'high' , 'volume', 'marketcap', 'trend']
predictor_variables = ['open', 'close', 'low', 'high' , 'volume']

n_predictor_variables = len(predictor_variables)



def create_data(price_dict, n_predictor_days, keywords=None, rel_train_size=0.9, scope='daily', 
        logger=None, verbose=False, return_single_dim=False, remove_outliers=True, 
        stocktype='crypto', minimum_rate=None, maximum_loss=None, leave_out_days=0,
        shuffle=True, mode='long'):
    if verbose:
        print('Creating training and test data.')
    limit = n_predictor_days * 2
    
    if type(price_dict) == str:
        price_dict = [price_dict]
    if type(price_dict) == list:
        # Its a list of strings, each corresponding to some ticker. Scrape data!
        price_dict_new = dict()
        for coin in price_dict:
            df = df_from_coin(coin, verbose=verbose, stocktype=stocktype)
            price_dict_new[coin] = df
        price_dict = price_dict_new


    data_packages = []
    dfs = []
    for i, (name, df) in enumerate(price_dict.items()):
        # Retrieve dataframe of coin price data
        try:
            if type(scope) == str:
                if scope == 'weekly':
                    print('Transforming daily to weekly data')
                    df = df_to_weekly(df)
                    print(f'there are {df.shape[0]} weeks in {name}')
            elif type(scope) == int:
                print('Transforming daily to 2-day data')
                df = df_to_weekly(df, weekdays=scope)
                print(f'there are {df.shape[0]} {scope}-days in {name}')

            if df.shape[0] < limit:
                if verbose:
                    print(f'{name} has less than {limit} days of data -> skipping that one!')
                continue

            x, train_days = get_train_test_data(df[:-int(leave_out_days)], predictor_variables, n_predictor_days=n_predictor_days,
                    return_single_dim=return_single_dim)

        except:
            # Coin not available or something went wrong
            if verbose:
                print(f'Something went wrong with coin {name}')
            continue
        
        if verbose:
            print(f'Scraped {name} ({i+1}/{len(price_dict)})')
        if verbose and i==0:
            print(f'\nLatest date scraped: {get_latest_day_from_df(df)}\n')

        if logger is not None:
            msg = f'Analysis is based on {scope} data.'
            logger.info(msg)
        
        data_packages.append( [x, train_days] )
        dfs.append(df)
    xy = np.concatenate([pack[0] for pack in data_packages], axis=0)

    if verbose:
        print(f'Final data set contains {xy.shape[0]} units of {scope} data points.')

    ## Norm without target
    x, y = unpack_x_y(xy, n_predictor_days, return_single_dim=return_single_dim, minimum_rate=minimum_rate, 
        maximum_loss=maximum_loss, mode=mode)
    if verbose:
        print(f'Normalizing')
    x_normed = normalize_candle_volume_marketcap(x, n_predictor_days, flatten=return_single_dim)

    if verbose:
        print(f'Remove Nans')
    
    # Remove Nans: 
    x_normed, y = remove_nan_rows(x_normed.astype(np.float64), y)
        
    # Shuffling
    indices = np.arange(y.shape[0])
    if shuffle:
        np.random.shuffle(indices)
    x_normed = x_normed[indices]
    y = y[indices]

    stop = int(round(y.shape[0]*rel_train_size))
    x_tr_normed = x_normed[:stop]
    y_tr = y[:stop]

    x_test_normed= x_normed[stop:]
    y_test = y[stop:]
        
    return x_tr_normed, y_tr, x_test_normed, y_test, dfs


# def create_data(coins, n_predictor_days, keywords=None, rel_train_size=0.9, scope='weekly', 
#         logger=None, verbose=False, return_single_dim=False, remove_outliers=True, 
#         stocktype='crypto'):

#     if type(coins) == str:
#         coins = [coins]
#     if verbose:
#         print('Creating training and test data.')
#     limit = n_predictor_days * 2
    
#     dfs = []
#     data_packages = []
#     for i, coin in enumerate(coins):
#         # Retrieve dataframe of coin price data
#         try:
#             if keywords is not None:
#                 df = df_from_coin(coin, keywords=keywords[i], verbose=verbose, stocktype=stocktype)
#             else:
#                 df = df_from_coin(coin, verbose=verbose, stocktype=stocktype)

#             if type(scope) == str:
#                 if scope == 'weekly':
#                     print('Transforming daily to weekly data')
#                     df = df_to_weekly(df)
#                     print(f'there are {df.shape[0]} weeks in {coin}')
#             elif type(scope) == int:
#                 print('Transforming daily to 2-day data')
#                 df = df_to_weekly(df, weekdays=scope)
#                 print(f'there are {df.shape[0]} {scope}-days in {coin}')

#             if df.shape[0] < limit:
#                 if verbose:
#                     print(f'{coin} has less than {limit} days of data -> skipping that one!')
#                 continue
#             x_tr, x_test, train_days, test_days = get_train_test_data(df, predictor_variables, n_predictor_days=n_predictor_days,
#                 rel_train_size=rel_train_size, return_single_dim=return_single_dim)
#         except:
#             # Coin not available or something went wrong
#             print(f'Something went wrong with coin {coin}')
#             continue
#         print(f'Scraped {coin} ({i+1}/{len(coins)})')
#         if i==0:
#             print(f'\nLatest date scraped: {get_latest_day_from_df(df)}\n')

#         if logger is not None:
#             msg = f'Analysis is based on {scope} data.'
#             logger.info(msg)
        
        

#         data_packages.append( [x_tr, x_test, train_days, test_days] )
#         dfs.append(df)
#     xy_tr = np.concatenate([pack[0] for pack in data_packages], axis=0)
#     xy_test = np.concatenate([pack[1] for pack in data_packages], axis=0)


#     if verbose:
#         print(f'Final data set contains {xy_tr.shape[0] + xy_test.shape[0]} units of {scope} data points.')

#     ## Norm without target
#     x_tr, x_test, y_tr, y_test = unpack_x_y(xy_tr, xy_test, n_predictor_days,
#                     return_single_dim=return_single_dim)
#     if verbose:
#         print(f'Normalizing')
#     x_tr_normed = normalize_candle_volume_marketcap(x_tr, n_predictor_days, flatten=return_single_dim)
#     x_test_normed = normalize_candle_volume_marketcap(x_test, n_predictor_days, flatten=return_single_dim)

#     if verbose:
#         print(f'Remove Nans')
    
#     # Remove Nans: 
#     x_tr_normed, y_tr = remove_nan_rows(x_tr_normed.astype(np.float64), y_tr)
#     x_test_normed, y_test = remove_nan_rows(x_test_normed.astype(np.float64), y_test)
    
#     # Balance Classes:
#     # x_tr_normed, y_tr = balance_class(x_tr_normed, y_tr)
#     # x_test_normed, y_test = balance_class(x_test_normed, y_test)
#     x = np.concatenate( [x_tr_normed, x_test_normed], axis=0 )
#     y = np.concatenate( [y_tr, y_test], axis=0 )

#     # x, y = balance_class(x, y.astype(bool))
    
#     # Shuffling
#     indices = np.arange(y.shape[0])
#     np.random.shuffle(indices)
#     x = x[indices]
#     y = y[indices]

#     stop = int(round(y.shape[0]*rel_train_size))
#     x_tr_normed = x[:stop]
#     y_tr = y[:stop]

#     x_test_normed= x[stop:]
#     y_test = y[stop:]
        
#     return x_tr_normed, y_tr, x_test_normed, y_test, dfs


def df_from_coin(coin, keywords=None, verbose=False, stocktype='crypto'):
    if verbose:
        print(f'Scrape historical {coin} prices...')

    if stocktype == 'crypto':
        scraper = CmcScraper(coin)
        df = scraper.get_dataframe()    
    elif stocktype == 'stock' or stocktype == 'stonk':
        df = yf.download(coin, verbose=0)
        df['time'] = df.index
        df.index = np.arange(df.shape[0])
        df = df.drop(['Adj Close'], axis=1)
    
    df = df.rename(columns={"Date": "time", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume", "Market Cap": "marketcap"})

    if 'marketcap' in df.columns:
        df = df[['time', 'open', 'close', 'low', 'high', 'volume', 'marketcap']]
    else:
        df = df[['time', 'open', 'close', 'low', 'high', 'volume']]

    df = df.sort_values('time')
    df.index = np.arange(df.shape[0])

    if keywords is not None:
        if verbose:
            print(f'Scrape historical {coin} trends...')
        df_trends = get_daily_trend(keywords, initial_date=pd.to_datetime(str(df['time'][0])[:10]))
        df = trends_to_coin(df, df_trends)

    return df

def get_latest_day_from_df(df):
    time = df['time']
    date = time.values[-1]
    return date
    

def get_lstm_model():

    act = 'swish'
    model = Sequential()
    model.add(LSTM(100, return_sequences=True))
    model.add(LSTM(100, return_sequences=True))
    model.add(LSTM(100))

    model.add(Dense(200, activation=act))
    model.add(Dense(200, activation=act))
    
    model.add(Dense(1, activation='linear'))
    
    model.compile(loss=loss, optimizer=optimizer)
    
    # if verbose:
    #     model.summary()
    return model

# OLD FUNCTION:
# def train_lstm_model(model, x_tr, x_test, x_tr_normed, x_test_normed, n_predictor_days, batch_size=128,
#     epochs=200, shuffle=True, validation_split=0.8, patience=25, logger=None, eval_only=False,
#     verbose=False):

#     tf.keras.backend.set_image_data_format('channels_last')

#     long_shape_tr = (x_tr_normed.shape[0], n_predictor_days+1, n_predictor_variables)
#     #short_shape_tr = (x_tr_normed.shape[0], np.prod((n_predictor_days, n_predictor_variables)))

#     long_shape_test = (x_test_normed.shape[0], n_predictor_days+1, n_predictor_variables)
#     #short_shape_test = (x_test_normed.shape[0], np.prod((n_predictor_days, n_predictor_variables)))
    

#     x_tr_tmp = x_tr_normed.reshape(*long_shape_tr)[:, :-1, :]
#     y_tr_tmp = ((x_tr.reshape(*long_shape_tr)[:, -1, 1] / x_tr.reshape(*long_shape_tr)[:, -1, 0]) - 1) * 100
#     y_tr_tmp[np.isnan(y_tr_tmp)] = 0
#     #x_tr_tmp = x_tr_tmp.reshape(*short_shape_tr)

#     x_test_tmp = x_test_normed.reshape(*long_shape_test)[:, :-1, :]
#     y_test_tmp = ((x_test.reshape(*long_shape_test)[:, -1, 1] / x_test.reshape(*long_shape_test)[:, -1, 0]) - 1) * 100
#     y_test_tmp[np.isnan(y_test_tmp)] = 0
#     #x_test_tmp = x_test_tmp.reshape(*short_shape_test)
    
#     if verbose:
#         print(f'Training with {x_tr_tmp.shape[0]} data samples (=days)')
#         print(f'Testing with {x_test_tmp.shape[0]} data samples (=days)')
#     print(f'x_tr_tmp[0]={x_tr_tmp[0]}\n')
#     print(f'y_tr_tmp[0]={y_tr_tmp[0]}\n')
#     if not eval_only:
#         es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
#         model.fit(x=x_tr_tmp, y=y_tr_tmp, validation_split=validation_split, batch_size=batch_size, epochs=epochs, 
#             shuffle=shuffle, callbacks=[es], verbose=1)
    
#     # Evaluate Test Loss of Model
#     test_loss = model.evaluate(x_test_tmp, y_test_tmp)
    
#     # Some Logging
#     msg = f'Test loss: {test_loss:.2f}'
#     if logger is not None:
#         logger.info(msg)
#     if verbose:
#         print(msg)

#     return model

# NEW FUNCTION

def train_lstm_model(model, x_tr_normed, y_tr, x_test_normed, y_test, n_predictor_days, batch_size=128,
    epochs=200, shuffle=True, validation_split=0.9, patience=25, logger=None, eval_only=False,
    verbose=False):

    tf.keras.backend.set_image_data_format('channels_last')

    long_shape_tr = (x_tr_normed.shape[0], n_predictor_days, n_predictor_variables)
    long_shape_test = (x_test_normed.shape[0], n_predictor_days, n_predictor_variables)
    

    x_tr_long = x_tr_normed.reshape(*long_shape_tr)
    x_test_long = x_test_normed.reshape(*long_shape_test)

    
    if verbose:
        print(f'Training with {x_tr_normed.shape[0]} data samples (=days)')
        print(f'Testing with {x_test_normed.shape[0]} data samples (=days)')

    # print(f'x_tr_normed[0]={x_tr_normed[0]}\n')
    # print(f'y_tr[0]={y_tr[0]}\n')

    if not eval_only:
        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
        model.fit(x=x_tr_long, y=y_tr, validation_split=validation_split, batch_size=batch_size, epochs=epochs, 
            shuffle=shuffle, callbacks=[es], verbose=1)
    
    # Evaluate Test Loss of Model
    test_loss = model.evaluate(x_test_long, y_test)
    
    # Some Logging
    msg = f'Test loss: {test_loss:.4f}'
    if logger is not None:
        logger.info(msg)
    if verbose:
        print(msg)

    return model

def get_ann_model(n_predictor_days=8, verbose=False):

    n_input_neurons = n_predictor_days*n_predictor_variables
    act = 'swish'

    model = Sequential()

    model.add(Dense(int(n_input_neurons/2), input_dim=int(n_input_neurons), activation=act))
    model.add(Dense(int(n_input_neurons/4), activation=act))
    model.add(Dense(int(n_input_neurons/8), activation=act))
    model.add(Dense(1, activation='linear'))

    
    model.compile(optimizer=optimizer, loss=loss)
    if verbose:
        model.summary()
    return model

def train_ann_model(model, x_tr_normed, y_tr, x_test_normed, y_test, n_predictor_days, batch_size=128,
    epochs=200, shuffle=True, validation_split=0.8, patience=25, logger=None, eval_only=False,
    verbose=False):


    long_shape_tr = (x_tr_normed.shape[0], n_predictor_days, n_predictor_variables)
    long_shape_test = (x_test_normed.shape[0], n_predictor_days, n_predictor_variables)

    short_shape_tr = (x_tr_normed.shape[0], np.prod((n_predictor_days, n_predictor_variables)))
    short_shape_test = (x_test_normed.shape[0], np.prod((n_predictor_days, n_predictor_variables)))

    x_tr = x_tr_normed.reshape(*short_shape_tr)    
    x_test = x_test_normed.reshape(*short_shape_test)
    # x_tr_tmp = x_tr_normed.reshape(*long_shape_tr)[:, :-1, :]
    # y_tr_tmp = ((x_tr.reshape(*long_shape_tr)[:, -1, 1] / x_tr.reshape(*long_shape_tr)[:, -1, 0]) - 1) * 100
    # y_tr_tmp[np.isnan(y_tr_tmp)] = 0
    # x_tr_tmp = x_tr_tmp.reshape(*short_shape_tr)

    # x_test_tmp = x_test_normed.reshape(*long_shape_test)[:, :-1, :]
    # y_test_tmp = ((x_test.reshape(*long_shape_test)[:, -1, 1] / x_test.reshape(*long_shape_test)[:, -1, 0]) - 1) * 100
    # y_test_tmp[np.isnan(y_test_tmp)] = 0
    # x_test_tmp = x_test_tmp.reshape(*short_shape_test)
    
    if verbose:
        print(f'Training with {x_tr.shape[0]} data samples (=days)')
        print(f'Testing with {x_test.shape[0]} data samples (=days)')
        
    if not eval_only:
        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
        model.fit(x=x_tr, y=y_tr, validation_split=validation_split, batch_size=batch_size, epochs=epochs, 
            shuffle=shuffle, callbacks=[es], verbose=1)
    
    # Evaluate Test Loss of Model
    test_loss = model.evaluate(x_test, y_test)
    
    # Some Logging
    msg = f'Test loss: {test_loss:.2f}'
    if logger is not None:
        logger.info(msg)
    if verbose:
        print(msg)

    return model

def predict_coins(coins, model, n_predictor_days, scope='daily', verbose=False, plotme=False, flatten=True):
    tf.keras.backend.set_image_data_format('channels_last')
    if type(coins) == str:
        coins = [coins]
    if verbose:
        print('\n==============================================')
        print('Predictions:\n\n')
    if type(scope) == str:
        if scope=='weekly':
            time_interval = 'week'
        elif scope=='daily':
            time_interval = 'day'
        else:
            msg = f'scope must be weekly or daily but is {scope}'
            raise ValueError(msg)
    elif type(scope) == int:
        time_interval = f'{scope}-day'
    else:
        msg = f'scope must be weekly or daily or some positive non-zero integer but is {scope}'
        raise ValueError(msg)

    prediction_dict = dict()
    for coin in coins:
        df = df_from_coin(coin)
        candlesticks = [df[pv].values[-n_predictor_days:] for pv in predictor_variables]        

        #dat = np.stack(candlesticks, axis=1).flatten()

        dat = np.stack(candlesticks, axis=1)
        if flatten:
            dat = dat.flatten()

        if n_predictor_variables == 6:
            msg = f'{n_predictor_variables} predictor variables is currently frozen cause it needs a fix that includes the norm_separately function.'
            raise NotImplementedError(msg)
            # dat_scaled = normalize_candle_volume_marketcap(np.expand_dims(dat, axis=0), n_predictor_days, flatten=flatten)[0]
        else:
            # dat_scaled = norm_separately(np.expand_dims(dat, axis=0))[0]
            dat_scaled = normalize_candle_volume_marketcap(np.expand_dims(dat, axis=0))[0]

        prediction = model.predict(np.expand_dims(dat_scaled, axis=0)[:4])[0][0]
        
        open_coin_normed = dat_scaled.reshape(n_predictor_days, n_predictor_variables)[-1, 1]
        predicted_perc_change = (prediction / open_coin_normed - 1) * 100
  
        open_coin = dat.reshape(n_predictor_days, n_predictor_variables)[-1, 1]

        close_coin = open_coin * (1 + predicted_perc_change / 100)

        

        prediction_dict[coin] = [open_coin, close_coin, predicted_perc_change]
        # is it going up (+) or down (-)
        if predicted_perc_change > 0:
            sign = '+'
        else:
            sign = '-'

        if verbose:
            print(f'\n{coin}: \tNext {time_interval} closes at price {close_coin:.4f} ({sign}{np.abs(predicted_perc_change):.2f} %)')
        
        if plotme:
            predicted_candle = [open_coin, close_coin, np.min([open_coin, close_coin]), np.max([open_coin, close_coin])]
            candles = np.stack(candlesticks[:4], axis=1)
            predictors_and_prediction = np.concatenate( (candles, np.expand_dims(predicted_candle, axis=0)), axis=0)
            title=f'{coin}: Predictors and prediction (last candle)'
            fig = go.Figure(data=[go.Candlestick(
                open=predictors_and_prediction[:, 0],
                high=predictors_and_prediction[:, 3],
                low=predictors_and_prediction[:, 2],
                close=predictors_and_prediction[:, 1],
                )])
            fig.update_layout(xaxis_rangeslider_visible=False, title=title)
            fig.show()

    if verbose:
        print('\n==============================================')
    return prediction_dict


def predict_coins_binary(coins, model, n_predictor_days, minimum_rate=None, maximum_loss=None, scope='daily', verbose=False, plotme=False, flatten=True,
        stocktype='crypto'):
    tf.keras.backend.set_image_data_format('channels_last')
    if type(coins) == str:
        coins = [coins]
    if verbose:
        print('\n==============================================')
        print('Predictions:\n\n')

    if scope=='weekly':
        time_interval = 'week'
    elif scope=='daily':
        time_interval = 'day'
    else:
        msg = f'scope must be weekly or daily but is {scope}'
        raise ValueError(msg)

    prediction_dict = dict()
    for i, coin in enumerate(coins):
        if verbose:
            print(f'Fetching {coin} ({i}/{len(coins)})')
        try:
        # df = df_from_coin(coin, stocktype=stocktype)

            x_tr, y_tr, x_test, y_test, dfs = create_data(coin, n_predictor_days, 
                keywords=None, verbose=verbose, scope=scope, return_single_dim=flatten, 
                stocktype=stocktype, minimum_rate=minimum_rate, maximum_loss=maximum_loss)
            df = dfs[0]
            # print(df)

            candlesticks = [df[pv].values[-n_predictor_days:] for pv in predictor_variables]        
            dat = np.stack(candlesticks, axis=1)
            

            if flatten:
                dat = dat.flatten()

            dat_scaled = normalize_candle_volume_marketcap(np.expand_dims(dat, axis=0), n_predictor_days, flatten=flatten)[0]

            # if not flatten:
            #     dat_scaled = dat_scaled.reshape(n_predictor_days, int(len(dat_scaled)/n_predictor_days))
            
            prediction = model.predict(np.expand_dims(dat_scaled, axis=0)[:4])[0][0]    

        except:
            continue

        if len(set(y_tr))>2 and minimum_rate is not None:
            y_tr = y_tr > minimum_rate
        if y_tr.sum()>0:
            x_tr, y_tr = balance_class(x_tr, y_tr)
        p = model.predict(x_tr)[:, 0]
        target_thresh, auc = get_threshold(y_tr, p)
        target_thresh = np.clip(target_thresh, a_min=0, a_max=1)
        accuracy = ((p>0.5)==y_tr).sum() / len(p)
        prediction_dict[coin] = dict(prediction=prediction, target_thresh=target_thresh, acc=accuracy, auc=auc, signal=prediction>=target_thresh)

    df_all = pd.DataFrame(prediction_dict).T.sort_values('prediction', ascending=False)
    df_go = df_all[df_all['signal']==True]
    df_go = df_go[df_go['auc']>0.5]
    return df_all, df_go

def get_threshold(y, p, target_precision=0.70):
    
    thr = 0.1
    precision = precision_score(y, p>thr)
    while precision<target_precision and thr<1:
        thr += 0.005
        precision = precision_score(y, p>thr, zero_division=0)
        
    return thr


def plot_candle_eval(model, x_test, y_test, n_predictor_days):

    long_shape_test = (x_test.shape[0], n_predictor_days+1, n_predictor_variables)
    short_shape_test = (x_test.shape[0], np.prod((n_predictor_days, n_predictor_variables)))

    sample = np.random.randint(0, x_test.shape[0])
    
    model_input = np.expand_dims(x_test[sample], axis=0)
    predicted_target_close = np.squeeze(model.predict(model_input))
    
    true_close = y_test[sample]
    last_day_close = x_test[sample, -1, 1]

    # predicted_target_close = predicted_target_close / last_day_close
    # true_target_close = true_close / last_day_close

    true_target_candle = np.expand_dims(np.array([
        last_day_close, 
        true_close, 
        np.min([last_day_close, true_close]), 
        np.max([last_day_close, true_close])
        ]), axis=0)

    predicted_target_candle = np.expand_dims(np.array([
        last_day_close, 
        predicted_target_close, 
        np.min([last_day_close, predicted_target_close]), 
        np.max([last_day_close, predicted_target_close])
        ]), axis=0)
    
    
    print(f'predicted_target_close={predicted_target_close}')
    print(f'true_close={true_close}')
   

    x_test_true = np.append(x_test[sample], true_target_candle, axis=0)
    fig = go.Figure(data=[go.Candlestick(
        open=x_test_true[:, 0],
        high=x_test_true[:, 3],
        low=x_test_true[:, 2],
        close=x_test_true[:, 1],
        )])
    fig.update_layout(xaxis_rangeslider_visible=False)
    fig.show()

    x_test_predicted = np.append(x_test[sample], predicted_target_candle, axis=0)
    fig = go.Figure(data=[go.Candlestick(
        open=x_test_predicted[:, 0],
        high=x_test_predicted[:, 3],
        low=x_test_predicted[:, 2],
        close=x_test_predicted[:, 1],
        )])
    fig.update_layout(xaxis_rangeslider_visible=False)
    fig.show()

# OLD FUNCTION:
# def plot_candle_eval(model, x_test, x_test_normed, n_predictor_days):

#     long_shape_test = (x_test_normed.shape[0], n_predictor_days+1, n_predictor_variables)
#     short_shape_test = (x_test_normed.shape[0], np.prod((n_predictor_days, n_predictor_variables)))

#     x_test_tmp = x_test_normed.reshape(*long_shape_test)[:, :-1, :]
#     y_test_tmp = ((x_test.reshape(*long_shape_test)[:, -1, 1] / x_test.reshape(*long_shape_test)[:, -1, 0]) - 1) * 100
#     y_test_tmp[np.isnan(y_test_tmp)] = 0
#     x_test_tmp = x_test_tmp.reshape(*short_shape_test)

#     sample = np.random.randint(0, x_test_normed.shape[0])
#     model_input = np.expand_dims(x_test_tmp[sample], axis=0)
#     prediction = np.squeeze(model.predict(model_input))
    
#     open_coin = x_test.reshape(*long_shape_test)[sample, -1, 0]
#     close_coin = open_coin * (1 + prediction / 100)

#     print(f'prediction={prediction}')
#     print(f'open_coin={open_coin}')
#     print(f'close_coin={close_coin}')

#     predicted_candle = [open_coin, close_coin, np.min([open_coin, close_coin]), np.max([open_coin, close_coin])]
#     print(f'predicted_candle={predicted_candle}')
#     x_test_long = x_test.reshape(x_test.shape[0], n_predictor_days+1, n_predictor_variables)
#     y_test_normed_reshape = y_test_tmp


#     fig = go.Figure(data=[go.Candlestick(
#         open=x_test_long[sample, :, 0],
#         high=x_test_long[sample, :, 3],
#         low=x_test_long[sample, :, 2],
#         close=x_test_long[sample, :, 1],
#         )])
#     fig.update_layout(xaxis_rangeslider_visible=False)
#     fig.show()

#     predictors_and_prediction = np.concatenate( (x_test_long[sample, :-1, :], np.expand_dims(predicted_candle, axis=0)), axis=0)
#     fig = go.Figure(data=[go.Candlestick(
#         open=predictors_and_prediction[:, 0],
#         high=predictors_and_prediction[:, 3],
#         low=predictors_and_prediction[:, 2],
#         close=predictors_and_prediction[:, 1],
#         )])
#     fig.update_layout(xaxis_rangeslider_visible=False)
#     fig.show()

def get_logger(scope):
    today = datetime.today().strftime('%Y-%m-%d')
    filename = f'C:/Users/Lukas/Documents/projects/stonks/logs/{today}_{scope}.log'
    # logging.basicConfig( level=logging.DEBUG, filename=filename)
    logger = logging.getLogger('The Log')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(filename)
    logger.addHandler(fh)
    msg = f'Price report {today}'
    logger.info(msg)
    return logger

def log_price(logger, price_dict):
    for key in price_dict.keys():
        open_price = price_dict[key][0]
        close_price = price_dict[key][1]
        change_perc = price_dict[key][2]
        if change_perc<0:
            sign = '-'
        else:
            sign = '+'
            
        msg = f'{key}: Open: {open_price:.4f}\tClose: {close_price:.4f}\tPerc. Change: ({sign}{np.abs(change_perc):.1f} %)'
        logger.info(msg)
    

def save_model(model, fn='assets/model/'):
    filepath = os.getcwd()
    os.chdir(fn)
    
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")

    print("Saved model to disk")
    os.chdir(filepath)

def load_model(pth='assets/model/', verbose=0):
    tf.keras.backend.set_image_data_format('channels_last')

    # load json and create model
    json_file = open(pth + '/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = tf.keras.models.model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights(pth  + "/model.h5")
    
    if verbose == 1:
        print("Loaded model from disk")
        model.summary()
    model.compile(optimizer=optimizer, loss=loss)
    return model

def df_to_weekly(df, weekdays=7):
    
    days = df.shape[0]
    residual = np.mod(days, weekdays)
    actual_days = days -  np.mod(days, weekdays)
    weeks = np.arange(residual, actual_days+1+residual, weekdays)
    n_weeks = len(weeks)

    time_cols = ['timestart', 'timestop']
    df_weekly = pd.DataFrame(columns=time_cols + list(df.columns))
    print(df_weekly)

    for week in range(n_weeks-1):

        df_week = df[weeks[week]:weeks[week+1]]
        w_time_start = str(pd.to_datetime(df_week['time']).dt.date.values[0])
        w_time_stop = str(pd.to_datetime(df_week['time']).dt.date.values[-1])
        
        w_time =  w_time_start + ' - ' + w_time_stop
        w_open = df_week['open'].values[0]
        w_high = df_week['high'].values.max()
        w_low = df_week['low'].values.min()
        w_close = df_week['close'].values[-1]
        w_volume = df_week['volume'].values.sum()
        try:
            w_marketcap = df_week['marketcap'].mean()
        except:
            w_marketcap = 0
        try:
            w_trend = df_week['trend'].mean()
        except:
            w_trend = 0
        if 'marketcap' in df.columns and 'trend' in df.columns:
            df_weekly.loc[week] = [w_time_start, w_time_stop, w_time, w_open, w_high, w_low, w_close, w_volume, w_marketcap, w_trend]
        elif 'marketcap' in df.columns:
            df_weekly.loc[week] = [w_time_start, w_time_stop, w_time, w_open, w_high, w_low, w_close, w_volume, w_marketcap]
        else:
            df_weekly.loc[week] = [w_time_start, w_time_stop, w_time, w_open, w_high, w_low, w_close, w_volume]
        

    return df_weekly

def scope_word(scope):
    if type(scope) == str:
        if scope == 'weekly':
            out = 'week'
        elif scope == 'daily':
            out = 'day'
    elif type(scope) == int:
        out = f'{scope}-day'
    else:
        msg = f'scope must be weekly, daily or some positive integer; but is {scope} '
        raise ValueError(msg)

    return out

def basic_log(coin, logger, df, n_predictor_days, scope):
    logger.info('\n')
    logger.info('====================================')
    logger.info(coin)
    startdate, enddate = [df[0]['time'].values[0], df[0]['time'].values[-1]]
    date_msg = f'Training data ranged from {startdate} to {enddate}'
    logger.info(date_msg)
    predictor_startdate, predictor_enddate = [df[0]['time'].values[-n_predictor_days], df[0]['time'].values[-1]]
    prediction_date_msg = f'Prediction for next {scope_word(scope)} is based on last {n_predictor_days} {scope_word(scope)}s: {predictor_startdate} to {predictor_enddate}'

    logger.info(prediction_date_msg)


def model_eval(model, x_test, y_test):
    predictions = np.squeeze(model.predict(x_test))
    if len(predictions.shape) == 0:
        return np.inf, np.inf, np.inf
    rel_errors = np.zeros((predictions.shape[0]))
    for i, (prediction, x, y) in enumerate(zip(predictions, x_test, y_test)):
        open_target = x[-1, 1]
        close_target_true = y
        close_target_predicted = prediction

        predicted_perc_change =  (close_target_predicted / open_target - 1) * 100
        true_perc_change =  (close_target_true / open_target - 1) * 100
        
        rel_errors[i] = predicted_perc_change - true_perc_change

    median_abs_error = np.median(np.abs(rel_errors))
    mad = np.mean(np.abs(rel_errors) - median_abs_error)
    low = np.percentile(np.abs(rel_errors), 5)
    high = np.percentile(np.abs(rel_errors), 95)

    print(f'median_abs_error: {median_abs_error:.2f} % +- {mad:.2f} %')
    print(f'Central 90% of errors lie between: {low:.2f} - {high:.2f} %')

    sns.displot(np.abs(rel_errors), kind="kde")
    plt.title('Absolute Error Distribution (based on Test Set)')
    plt.xlabel(f'Absolute error percentage [%]')

    return median_abs_error, low, high


def unpack_x_y(x, n_predictor_days, return_single_dim=True, minimum_rate=None, maximum_loss=None, mode='long'):
    # OCLH
    long_shape = (x.shape[0], n_predictor_days+1, n_predictor_variables)
    short_shape = (x.shape[0], np.prod((n_predictor_days, n_predictor_variables)))

    

    x_long = x.reshape(*long_shape)[:, :-1, :]

    y = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        open_price = x[i, -1, 0]
        close_price = x[i, -1, 1]
        low_price = x[i, -1, 2]
        high_price = x[i, -1, 3]
        
        if minimum_rate is None and maximum_loss is None:
            if open_price == 0:
                y[i] = 1
                continue
            y[i] = close_price / open_price

        elif minimum_rate is not None and maximum_loss is not None:
            if open_price == 0:
                y[i] = 0
                continue
            
            if mode=='long':
                up = high_price / open_price
                down = low_price / open_price
                if up > minimum_rate and down > maximum_loss:
                    y[i] = 1
                else:
                    y[i] = 0
            elif mode=='short':
                down = low_price / open_price
                up = high_price / open_price
                if down < minimum_rate and up < maximum_loss:
                    y[i] = 1
                else:
                    y[i] = 0
        else:
            msg = f'Please set both minimum_rate and maximum_loss OR set both to None!'
            raise ValueError(msg)
            
        
    if return_single_dim:
        x_long = x_long.reshape(*short_shape)
        
    return x_long, y

def central_99(x, y):
    ''' Calculate the percentage change of last predictor days closing price to
    target day closing price and select only those samples that lie in the
    central 90 % of percentage change'''

    perc_change = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        target_day_close = y[i]
        last_day_close = x[i, -1, 1]
        perc_change[i] = target_day_close / last_day_close 

    lower = np.percentile(perc_change, 0.5)
    upper = np.percentile(perc_change, 99.5)
    vals_in_range = (perc_change>lower) & (perc_change<upper)

    return x[vals_in_range], y[vals_in_range]

class MetaModel:
    def __init__(self, n_models=2, verbose=True):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

        self.n_models = n_models
        self.models = [self.get_lstm_model(verbose=verbose) for _ in range(n_models)]
        self.models += self.get_linear_models()

        self.reg = linear_model.LinearRegression()

    def get_ensemble(self, x):
        lstm_predictions = [model.predict(x) for model in self.models[:self.n_models]]
        lstm_predictions = np.squeeze(np.stack(lstm_predictions, axis=1))

        x_flat = self.flatten(x)

        linear_predictions = []
        for model in self.models[self.n_models:]:
            if hasattr(model, 'predict_proba'):
                idx = np.where(model.classes_==1)[0][0]
                linear_predictions.append( model.predict_proba(x_flat)[:, idx] )
            else:
                linear_predictions.append( model.predict(x_flat) )

        linear_predictions = np.stack(linear_predictions, axis=0).T

        if len(linear_predictions.shape) == 1:
            linear_predictions = np.expand_dims(linear_predictions, axis=0)
        
        if len(lstm_predictions.shape) == 1:
            lstm_predictions = np.expand_dims(lstm_predictions, axis=0)
            
        ensemble_predictions = np.concatenate([lstm_predictions, linear_predictions], axis=1)

        return ensemble_predictions

    def train(self, x, y, verbose=True):        
        
        if verbose:
            print(f'Train {self.n_models} LSTMs and several linear models.')
        self.train_ensemble(x, y, verbose=verbose)
        
        if verbose:
            print(f'Getting ensemble predictions...')
        ensemble_predictions = self.get_ensemble(x)
        
        if verbose:
            print(f'Train ensemble classifier...')
        
        self.reg.fit(ensemble_predictions, y)
    
    def flatten(self, x):
        return x.reshape(x.shape[0], np.prod((x.shape[1], x.shape[2])))

    def train_ensemble(self, x_tr, y_tr, verbose=True):
        if verbose:
            print('Training LSTMs')
        models = []
        for i, model in enumerate(self.models[:self.n_models]):
            model.fit(x=x_tr, y=y_tr, validation_split=0.9, batch_size=256, epochs=1, 
                shuffle=True, callbacks=[self.es, self.rl], verbose=0)
            models.append(model)

        if verbose:
            print('Training linear models')
        
        x_tr_flat = self.flatten(x_tr)
        for i, model in enumerate(self.models[self.n_models:]):
            model.fit(x_tr_flat, y_tr)
            models.append(model)

        self.models = models
        
    def predict(self, x):
        if len(x.shape) < 3:
            x = np.expand_dims(x, 0)

        ensemble_predictions = self.get_ensemble(x)

        if hasattr(self.reg, 'predict_proba'):
            idx = np.where(self.reg.classes_==1)[0][0]
            p = self.reg.predict_proba(ensemble_predictions)[:, idx]
        else:
            p = np.clip(self.reg.predict(ensemble_predictions), a_min=0, a_max=1)

        return np.expand_dims(p, axis=1)

    def evaluate(self, x_test, y_test):
        p = self.predict(x_test)[:, 0]
        auc = roc_auc_score(y_test, p)
        return auc

    def get_lstm_model(self, verbose=True):
        tf.keras.backend.set_image_data_format('channels_last')
        patience = 15
        

        model = Sequential()
        model.add(Bidirectional(LSTM(10)))
        model.add(Dropout(0.2))
        model.add(Dense(10, activation='swish'))

        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), metrics=['accuracy'])

        self.es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
        self.rl = tf.keras.callbacks.ReduceLROnPlateau(min_lr=1e-6, verbose=1, patience=10)
        
        return model
    
    def get_linear_models(self):
        ''' Returns a list of linear models to to regression/classification with'''
        n_jobs=-1
        linear_models = [
            linear_model.LinearRegression(n_jobs=n_jobs),
            linear_model.Lasso(alpha=0.1),
            linear_model.BayesianRidge(),
            linear_model.LogisticRegression(n_jobs=n_jobs),
            linear_model.SGDClassifier(n_jobs=n_jobs),
            RandomForestClassifier(n_estimators=100, n_jobs=n_jobs),
        ]
        return linear_models

def get_model(x_tr, y_tr, verbose=True):
    tf.keras.backend.set_image_data_format('channels_last')

    shuffle = True
    batch_size = 256
    epochs = 100
    patience = 15

    model = get_lstm_model()
    act = 'swish'

    model = Sequential()
    # model.add(LSTM(100, return_sequences=True))
    model.add(Bidirectional(LSTM(20, return_sequences=True)))
    model.add(Dropout(0.4))
    model.add(Bidirectional(LSTM(20)))
    model.add(Dropout(0.4))

    # model.add(Dense(50, activation=act))
    model.add(Dense(10, activation=act))


    # model.add(Dense(1, activation='linear'))
    # model.compile(loss='mean_absolute_error', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001))

    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), metrics=['accuracy'])



    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
    rl = tf.keras.callbacks.ReduceLROnPlateau(min_lr=1e-6, verbose=1, patience=10)

    history = model.fit(x=x_tr, y=y_tr, validation_split=0.90, batch_size=batch_size, epochs=epochs, 
        shuffle=shuffle, callbacks=[es, rl], verbose=1)
    # if verbose:
    #     print(f'Test acc: {model.evaluate(x_test, y_test)[1]*100:.2f}%')

    # %matplotlib qt
    # plt.figure()
    # plt.subplot(211)
    # plt.plot(history.history['loss'], label='Training Loss')
    # plt.plot(history.history['val_loss'], label='Validation Loss')
    # plt.legend()
    # plt.subplot(212)
    # plt.plot(history.history['accuracy'], label='Training Accuracy')
    # plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    return model


def scrape_coins(symbols, verbose=True, stocktype='crypto'):
    coin_price_dict = dict()
    for i, symbol in enumerate(symbols):
        try:
            print(f'Scraping {symbol} ({i}/{len(symbols)})')
            coin_price_dict[symbol] = df_from_coin(symbol, verbose=verbose, stocktype=stocktype)
        except:
            print(f'Skipping {symbol}')
            pass
    return coin_price_dict
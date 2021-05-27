import datetime
import numpy as np
from copy import deepcopy
import pandas as pd
from sklearn.preprocessing import normalize as normalize_sk

def strip_date(stock, start_date=None, end_date=None):
    if start_date is not None:
        if '/' in start_date:
            start_date = datetime.datetime.strptime(start_date, '%m/%d/%Y')
        elif '-' in start_date:
            start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
            start_date = start_date.strftime('%m/%d/%Y')
        mask = stock.index > start_date
        stock = stock.loc[mask]
    if end_date is not None:
        if '/' in end_date:
            end_date = datetime.datetime.strptime(end_date, '%m/%d/%Y')
        elif '-' in end_date:
            end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d')
            end_date = end_date.strftime('%m/%d/%Y')
        mask = stock.index <= end_date
        stock = stock.loc[mask]

    return stock

def get_days_from_str(interval):
    if type(interval) == str:
        if interval == 'daily':
            days = 1
        if interval == 'weekly':
            days = 7
        if interval == 'monthly':
            days = 30
        if interval == 'yearly':
            days = 365
    elif isinstance(interval, (int, float)):
        days = int(round(interval))
    else:
        msg = f'interval must be of type str, int or float. Got {type(interval)} instead.'
        raise ValueError(msg)
        return None

    return days

def check_interval(stock, days):
    if (stock.index[-1] - stock.index[0]).days < days*2:
        msg = f'Warning, your investment interval of {days} days is too long for the stock history ({(stock.index[-1] - stock.index[0]).days} days)'
        print(msg)
    

def get_train_test_data(df, predictor_variables, n_predictor_days=5, return_single_dim=True):
    n_days = df.shape[0]
    # np.random.shuffle(all_days)
    train_days = np.arange(n_predictor_days, n_days)


    data = np.stack( [df[category].values for category in predictor_variables], axis=1 )

    # Training Data
    x = []
    for i in train_days:
        x.append(data[i-n_predictor_days:i+1, :])
        
    x = np.stack(x, axis=0)
    
    if return_single_dim:
        return reshape_last_two_dims(x), train_days
    else:
        return x, train_days


def normalize_candle_volume_marketcap(X, n_predictor_days, flatten=True):
    if len(X.shape) == 2:
        n_params = int(X.shape[1] / n_predictor_days)
    elif len(X.shape) == 3:
        n_params = X.shape[-1]

    long_shape = (X.shape[0], n_predictor_days, n_params)
    short_shape = (X.shape[0], np.prod((n_predictor_days, n_params)))

    X_long = X.reshape(*long_shape)

    X_candles_normed = normalize(X_long[:, :, :4])
    X_rest = []
    for i in range(4, X_long.shape[-1]):
        X_rest.append(np.expand_dims(normalize(X_long[:, :, i]), axis=-1))
    
    # X_volume_normed = np.expand_dims(normalize(X_long[:, :, 4]), axis=-1)
    # X_marketcap_normed = np.expand_dims(normalize(X_long[:, :, 5]), axis=-1)
    # X_trend_normed = np.expand_dims(normalize(X_long[:, :, 6]), axis=-1)
    
    # x_all_together_normed = np.concatenate([
    #     X_candles_normed, 
    #     X_volume_normed, 
    #     X_marketcap_normed, 
    #     X_trend_normed
    #     ], axis=-1)
    x_all_together_normed = np.concatenate([
        X_candles_normed, 
        *X_rest
        ], axis=-1)

    if flatten:
        x_all_together_normed = x_all_together_normed.reshape(*short_shape)

    return x_all_together_normed

def normalize(data):
    data_2 = deepcopy(data)
    for i in range(data_2.shape[0]):
        data_2[i] = centering_normalizer(data_2[i])
    return data_2

def logarithmize(data):
    data_2 = np.clip(deepcopy(data), a_min=1e-6, a_max=np.inf)
    return np.log(data_2)
    # for i in range(data_2.shape[0]):
    #     data_2[i] = np.log(data_2[i])

    return data_2

def centering_normalizer(x, epsilon=1e-6):
    # M = np.mean(x)
    # scaler = np.max(np.abs((x-M)))
    # return (x-M) / np.clip(scaler, a_min=epsilon, a_max=1e20)
    # minmax = np.clip(x.max() - x.min(), a_min=epsilon, a_max=1e20)
    # return x/minmax
    return (x-x.mean()) / np.clip(x.std(), a_min=epsilon, a_max=np.inf)

def reshape_last_two_dims(x):
    return x.reshape(x.shape[0], np.prod(x.shape[1:3]))


def scale_prediction(prediction, X):
    M = np.mean(X)
    scaler = np.max(np.abs(X-M))
    out = prediction * scaler + M
    return out

def price_dicts_to_df(price_dicts):
    columns = ['Coin', 'Open', 'Close', 'Percentage Change [%]', 'Error', 'Error Low', 'Error High']
    df = pd.DataFrame(columns=columns)
    # coin = [i for i in price_dict.keys()][0]
    for i, d in enumerate(price_dicts):
        coin = [j for j in d.keys()][0]
        print(coin)
        open_coin = price_dicts[i][coin][0]
        close_coin = price_dicts[i][coin][1]
        perc_change = price_dicts[i][coin][2]
        lo = price_dicts[i][coin][3]
        hi = price_dicts[i][coin][4]
        error = price_dicts[i][coin][5]
        
        # perc_change_min = (np.abs(perc_change) - price_dicts[i][coin][3]) * np.sign(perc_change)


        # close_min_coin = open_coin * ((perc_change_min/100)+1)
        row = [coin, open_coin, close_coin, perc_change, error, lo, hi]
        df.loc[i] = row
    return df

def df_to_excel(df, name, scope):
    today = datetime.datetime.today().strftime('%Y-%m-%d')
    filename = f'{name}_{today}_{scope}.xlsx'
    filepath = 'C:/Users/Lukas/Documents/projects/stonks/assets/excel/'
    df.to_excel(filepath+filename, sheet_name=filename, index = False)

# def norm_separately(x, y=None):
#     n_samples, n_days, n_params = x.shape
#     x_norm, norms = normalize_sk(x.reshape(n_samples, np.prod((n_days, n_params))), return_norm=True, norm='max')
#     x_norm = x_norm.reshape(n_samples, n_days, n_params)
#     if y is not None:
#         y_norm = y / norms
#         return x_norm, y_norm
#     else:
#         return x_norm

def norm_separately(x, y=None):
    n_samples, n_days, n_params = x.shape
    x_norm, norms = norm_last_unit(x)
    if y is not None:
        y_norm = y / norms
        return x_norm, y_norm
    else:
        return x_norm

def norm_last_unit(x):
    x_norm = np.zeros(x.shape)
    norms = np.zeros(x.shape[0])
    for i, d in enumerate(x):
        norms[i] = d[-1, 1]
        x_norm[i] = d / norms[i]
    return x_norm, norms

def balance_class(x, y):
    n_positive = int(np.sum(y))
    n_negative = int(len(y) - np.sum(y))

    idc_positive = np.where(y==True)[0]
    idc_negative = np.where(y==False)[0]

    if n_positive > n_negative:
        keep_negative = idc_negative
        np.random.shuffle(idc_positive)
        keep_positive = idc_positive[:n_negative]
        
        keep_idc = np.concatenate((keep_negative, keep_positive))
        np.random.shuffle(keep_idc)
        x, y = x[keep_idc], y[keep_idc]

    elif n_positive < n_negative:
        keep_positive = idc_positive
        np.random.shuffle(idc_negative)
        keep_negative = idc_negative[:n_positive]

        keep_idc = np.concatenate((keep_negative, keep_positive))
        np.random.shuffle(keep_idc)
        x, y = x[keep_idc], y[keep_idc]

    return x, y

def remove_nan_rows(x, y):
    good_idx = np.argwhere(np.isnan(x))
    x = np.delete(x, list(set(good_idx[:, 0])), axis=0 )
    y = np.delete(y, list(set(good_idx[:, 0])), axis=0 )
    return x, y

def ex_stable(coins, stablecoins):
    for stablecoin in stablecoins:
        if stablecoin in coins:
            coins.pop(coins.index(stablecoin))
    return coins
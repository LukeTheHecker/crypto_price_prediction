import numpy as np
import pandas as pd
from ..cryptobot import (predictor_variables, coins_binance, normalize_candle_volume_marketcap, 
                        create_data, get_threshold, get_train_test_data, unpack_x_y, remove_nan_rows)
from sklearn.metrics import balanced_accuracy_score, precision_score, precision_recall_curve, roc_curve, roc_auc_score

import matplotlib.pyplot as plt
from ...config import klines_names

def bars_to_df(bars, klines_names):
    bars = np.stack(bars, axis=0)
    d = dict()
    for i in range(bars.shape[1]):
        d[klines_names[i]] = bars[:, i]

    df = pd.DataFrame(d)
    df['open_time'] = [str(pd.to_datetime(int(val), unit='ms')) for val in df['open_time'].values]
    df['close_time'] = [str(pd.to_datetime(int(val), unit='ms')) for val in df['close_time'].values]
    df = df.rename(columns={"open_time": "time"})
    df = df[['time']+predictor_variables]
    df = df.sort_values('time')
    for key in predictor_variables:
        df[key] = df[key].values.astype(float)

    return df

def get_binance_dfs(client, base_currency='USDT', coins=coins_binance):
    coins = list(set(coins))
    dfs = dict()
    for i, coin in enumerate(coins):
        print(f'{coin} ({i+1}/{len(coins)})')
        try:
            # Scrape newest data
            timestamp = client._get_earliest_valid_timestamp(coin+base_currency, '1d')
            bars = client.get_historical_klines(coin+base_currency, '1d', timestamp, limit=1000)
            # Transform to dataframe
            df = bars_to_df(bars, config.klines_names)
        except:
            continue
        dfs[coin] = df
    return dfs

def shortshape(x):
    return x.reshape(x.shape[0], x.shape[1]*x.shape[2])

def get_binance_predictions(client, model, dfs, n_predictor_days, minimum_rate, maximum_loss, leave_out_days=0, 
        days_ago=0, plotme=False, mode='long', thr=None, target_precision=0.6, auc_thresh=0.6):
    prediction_dict = dict()
    # coins = list(set(coins_binance))
    base_currency = 'USDT'
    for coin, df in dfs.items():
        try:
            # print(f'{coin}')
            if df.shape[0] < leave_out_days/2:
                # print('\ttoo few days in df for good evaluation, skipping this')
                continue
        

            # Get and prepare latest data
            if days_ago>0:
                data = np.stack([df[key].values[-n_predictor_days-days_ago:-int(days_ago)] for key in predictor_variables], axis=1)
            else:
                data = np.stack([df[key].values[-n_predictor_days-days_ago:] for key in predictor_variables], axis=1)
            data = normalize_candle_volume_marketcap(np.expand_dims(data, axis=0), n_predictor_days, flatten=False)
            # Perform prediction for next day
            prediction = model.predict_proba(shortshape(data))[0][1]  # [0][0]
            # Process historical data of current coin for quality control
            if days_ago>0:
                xy, train_days = get_train_test_data(df[-int(leave_out_days):-int(days_ago)], predictor_variables, n_predictor_days=n_predictor_days, return_single_dim=False)
            else:
                xy, train_days = get_train_test_data(df[-int(leave_out_days):], predictor_variables, n_predictor_days=n_predictor_days, return_single_dim=False)
            x_tr, y_tr = unpack_x_y(xy, n_predictor_days, return_single_dim=False, minimum_rate=minimum_rate, maximum_loss=maximum_loss, mode=mode)
            x_tr = normalize_candle_volume_marketcap(x_tr, n_predictor_days, flatten=False)
            x_tr, y_tr = remove_nan_rows(x_tr.astype(np.float64), y_tr)
            n_tp = len(y_tr[y_tr==True])

            if n_tp < 2:
                # print('\ttoo few positives for good evaluation, skipping this')
                continue
            
            # Predict historical data of given coin and get quality metrics
            p = model.predict_proba(shortshape(x_tr))[:, 1]
            target_thresh = get_threshold(y_tr, p, target_precision=target_precision)
            if thr is not None:
                target_thresh =  thr
            else:
                target_thresh = np.clip(target_thresh, a_min=0, a_max=1)
            tmp = df['time'].values[-1-days_ago]

            last_close = df['close'].values[-1-days_ago]
            last_date = df['time'].values[-1-days_ago]

            accuracy = accuracy_score(y_tr, p>target_thresh)
            auc = roc_auc_score(y_tr, p)
            auc_pred = (prediction + auc) / 2
            # precision = precision_score(y_tr, p>target_thresh)
            precision = precision_score(y_tr, p>target_thresh)
            recall = recall_score(y_tr, p>target_thresh)
            bal_acc = balanced_accuracy_score(y_tr, p>target_thresh)
            signal = (prediction>=target_thresh) and (auc>auc_thresh)
            predicted_day = df['time'].values[-int(days_ago)]
            # Get actual result
            if days_ago>0:
                next_close = df['close'].values[-int(days_ago)]
                next_open = df['open'].values[-int(days_ago)]
                next_high = df['high'].values[-int(days_ago)]
                next_low = df['low'].values[-int(days_ago)]
                
                if mode=='long':
                    if (next_high/next_open) > minimum_rate and (next_low/next_open) > maximum_loss:
                        actual = True
                    else:
                        actual = False
                elif mode == 'short':
                    if (next_low/next_open) < minimum_rate and (next_high/next_open) < maximum_loss:
                        actual = True
                    else:
                        actual = False

                # actual = y_tr[-days_ago]
            else:
                actual = None
            # Summarize prediction, signal and quality metrics in dictionary
            prediction_dict[coin] = dict(prediction=prediction, target_thresh=target_thresh, precision=precision, recall=recall,
                                        bal_acc=bal_acc, acc=accuracy, auc=auc, auc_pred=auc_pred, n_tp=n_tp, last_close=last_close, 
                                        last_date=last_date, predicted_day=predicted_day, signal=signal, actual=actual)
            if plotme:
                x_tr, y_tr = balance_class(x_tr, y_tr)
                p = model.predict_proba(shortshape(x_tr))[:, 1]
                plot_roc(y_tr, p, coin=coin)
        except:
            continue
    # Put all dictionaries in dataframe for easier visualization and sort it such that best coins are up
    df_all = pd.DataFrame(prediction_dict).T.sort_values('auc_pred', ascending=False)

    # Get a list of all buy-signal coins
    df_go = df_all[df_all['signal']==True]

    return df_all, df_go



def plot_prc(y, p):
    precision, recall, thresholds = precision_recall_curve(y, p)
    
    # convert to f score
    fscore = (2 * precision * recall) / (precision + recall)
    # locate the index of the largest f score
    ix = np.argmax(fscore)
    # print('Best Threshold=%f, F-Score=%.3f' % (thresholds[ix], fscore[ix]))

    # plot the roc curve for the model
    no_skill = len(y[y==1]) / len(y)
    plt.figure()
    plt.plot([0,1], [no_skill,no_skill], linestyle='--', label='No Skill')
    plt.plot(recall, precision, marker='.', label='Logistic')
    plt.scatter(recall[ix], precision[ix], marker='o', color='black', label='Best')
    # axis labels
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    # show the plot
    plt.show()


def get_auc_for_df(df, model, n_predictor_days, minimum_rate, maximum_loss, leave_out_days, mode):
    x, y, x2, y2, _ = create_data(dict(coin=df), n_predictor_days, verbose=False, minimum_rate=minimum_rate, maximum_loss=maximum_loss, leave_out_days=1, mode=mode, rel_train_size=0.5)
    x = np.concatenate([x, x2], axis=0)
    y = np.concatenate([y, y2], axis=0)
    
    if len(set(y)) > 2:
        y = y > minimum_rate


    p = model.predict_proba(shortshape(x))[:, 1]
    # p = model.predict(x)
    try:
        auc = roc_auc_score(y, p)
    except:
        auc = 0.5
    return auc

def get_precision(y, p):
    p = np.squeeze(p)
    y = np.squeeze(y)
    tp = (p[p==True] == y[p==True]).sum()
    fp = (y[p==True] != p[p==True]).sum()

    return tp / (tp + fp)


def plot_roc(y, p, coin='', target_precision=0.7):
    fpr, tpr, thr = roc_curve(y, p)
    auc = roc_auc_score(y, p)

    threshold = get_threshold(y, p, target_precision=target_precision)

    # auc = roc_auc_score(y, p)
    acc = ((p>threshold)==y).sum() / len(p)

    plt.figure()
    plt.plot(fpr, tpr, label='ROC')
    plt.plot(fpr, thr, label='threshold')
    plt.plot([0,1],[0,1],label='origin')
    plt.plot([threshold,threshold],[0,1],label='threshold')

    plt.legend()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.ylim(0,1)
    plt.xlim(0,1)

    plt.title(f'{coin}: Thr: {threshold:.2f}, AUC={auc:.2f}, ACC={acc:.2f}')
    plt.show()


def get_test_dfs(dfs, leave_out_days):
    dfs_test = dict()
    for key, val in dfs.items():
        if val.shape[0] < leave_out_days:
            continue
        dfs_test[key] = val[-leave_out_days:]
    return dfs_test
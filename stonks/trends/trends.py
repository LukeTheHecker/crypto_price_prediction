import time
from copy import deepcopy
import pandas as pd
from pytrends.request import TrendReq
import numpy as np

def get_daily_trend(kw, initial_date = pd.to_datetime('2013-01-01'), w_time=1):

    if type(kw) == str:
        kw = [kw]

    pytrend = TrendReq(timeout=100, proxies=['https://34.203.233.13:80',])
    startdate = deepcopy(initial_date)
    enddate = startdate + pd.DateOffset(months=1)
    today = pd.to_datetime("today")
    # Scrape first Month
    while True:
        try:
            pytrend.build_payload(kw_list=kw, timeframe=f'{str(startdate)[:10]} {str(enddate)[:10]}')
            df_trend_daily = pytrend.interest_over_time()
            break
        except:
            time.sleep(w_time)


    # Add more months iteratively until we are at today's date
    while startdate < today:
        
        if enddate > today:
            enddate = today

        while True:
            try:
                pytrend.build_payload(kw_list=kw, timeframe=f'{str(startdate)[:10]} {str(enddate)[:10]}')
                df_tmp = pytrend.interest_over_time()
                break
            except: 
                time.sleep(w_time)

        df_trend_daily = pd.concat([df_trend_daily, df_tmp])
        # Change start and enddate to the next month
        startdate = startdate + pd.DateOffset(months=1)
        enddate = startdate + pd.DateOffset(months=1)

    # Scrape monthly data
    while True:
        try:
            pytrend.build_payload(kw_list=kw, timeframe=f'{str(initial_date)[:10]} {str(today)[:10]}')
            df_trend_monthly = pytrend.interest_over_time()
            break
        except:
            time.sleep(w_time)
            pass

    # Assign the monthly data to the daily data frame
    monthlies = np.zeros(df_trend_daily.shape[0])
    for i in range(df_trend_daily.shape[0]):
        idx = arg_time(df_trend_monthly.index, df_trend_daily.index[i])
        monthlies[i] = df_trend_monthly[kw[0]].values[idx]

    # Correct Daily data using the monthly date
    df_trend_daily[kw[0]] = (monthlies/100) * df_trend_daily[kw[0]].values

    return df_trend_daily

def arg_time(date_list, date_of_interest):
    ''' Find index of date_of_interest in date_list'''
    return np.argmin(np.abs((date_list - date_of_interest).astype('timedelta64')))

def trends_to_coin(df_coin, df_trends):
    trend_array = np.zeros(df_coin.shape[0])
    for i in range(df_coin.shape[0]):
        time_of_interest = df_coin['time'].values[i]
        idx = arg_time(df_trends.index, time_of_interest)
        trend_array[i]= df_trends[df_trends.columns[0]].values[idx]
    df_coin['trend'] = trend_array
    return df_coin
'''
Functions that calculate the expected value for a given stonk.
'''
import datetime
import numpy as np
from copy import deepcopy
from ..util import get_days_from_str, strip_date, check_interval



def stock_yield(stock, interval='yearly', start_date=None, end_date=None):
    '''
    Parameters:
    -----------
    start_date : str, of format M/D/YYYY, can be None -> all data is selected
    end_date : str, of format M/D/YYYY, can be None -> all data is selected
    '''
    stock = deepcopy(stock)
    days = get_days_from_str(interval)
    
    stock = strip_date(stock, start_date=start_date, end_date=end_date)
    check_interval(stock, days)
    print(stock.index)
    # Data frame columns to lowercase
    stock.columns= stock.columns.str.lower()
    # Get temporal resolution of stock
    delta = [stock.index[i+1] - stock.index[i] for i in range(10)]
    stock_interval_days = np.min([d.days for d in delta])
    
    points_per_interval = stock_interval_days*days
    if points_per_interval < 2:
        msg = f'Interval of {days} days is short since the stock contains only one data point per {stock_interval_days} days'
        raise ValueError(msg)
    
 
    stock_yields = list()

    for i in range(stock.shape[0]):
        buy_time = stock.index[i]
        sell_time = buy_time + datetime.timedelta(days)
        if (sell_time - buy_time).days < days-5:
            print(f'only {(sell_time - buy_time).days} between buy and sell - we are done!')
            break
        sell = stock.iloc[stock.index.get_loc(sell_time, method='nearest')]['close']
        buy = stock['close'][0]
        stock_yields.append(sell / buy)

    return np.array(stock_yields)
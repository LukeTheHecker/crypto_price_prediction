# Binance buy strategy

## Model preparation
* load dataframes of historical price data
* preprocess data (cut, normalize)
* Train model on data
* Assert that model has decent test-accuracy and ROC-AUC 
* log model test performance
  
## Buy preparation
In this step we aim to find the best coin to invest in today.

Start this at 11 PM:

For each coin:
* download data of all coins of interest from the past 30 days, including the unfinished day today
* Prepare data (normalize!)
* Predict next days candle

Find the best coin based on predicted value, AUC and Signal (sort the DataFrame)

## Order management

### How much to invest:
* Get your USDT Balance and reserve 10% of that for trading

### Order type(s)

Now we have various choices. 

1. Buy coin with market order, i.e. we just buy once we have the signal; regardless of the exact price. Then we set a stop loss of -5% in case our prediction goes downways. We also have to make sure we sell at the correct price. Therefore, we could set a take profit order at +10% or some trailing stop loss at -5%.
  

First of all, we believe price goes up that day. Therefore, our 



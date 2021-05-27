import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from stonks.cryptobot import *
from stonks.util import price_dicts_to_df, df_to_excel

# Settings
coins_buy = coins_coinbase
# coins_buy = ['BTC', 'ETH']

# coins_pretrain = coins_toponehundred
# coins_pretrain = ['BTC']

coins_train = coins_coinbase
# coins_train = ['BTC']

verbose = True
plotme = False
n_predictor_days = 90
scope = 'daily'
# n_predictor_days = 10
# scope = 'weekly'
batch_size = 256

# Logging
logger = get_logger(scope)
logger.info(f'Scope: {scope}, Predictor range: {n_predictor_days} {scope_word(scope)}s')

# Load LSTM Model
base_model = get_lstm_model()

# ================================================== #
# Pre-train Model with many many coins:
# x_tr, y_tr, x_test, y_test, _ = create_data(coins_pretrain, n_predictor_days, 
#                 verbose=verbose, scope=scope, return_single_dim=False)

# base_model = train_lstm_model(base_model, x_tr, y_tr, x_test, y_test, n_predictor_days, 
#                 logger=logger, batch_size=batch_size, verbose=verbose)


# # ================================================== #
# Re-train the pretrained model with the Coinbase set
x_tr, y_tr, x_test, y_test, _ = create_data(coins_train, n_predictor_days, 
                verbose=verbose, scope=scope, return_single_dim=False)

base_model = train_lstm_model(base_model, x_tr, y_tr, x_test, y_test, n_predictor_days, 
                logger=logger, batch_size=batch_size, verbose=verbose)

# ================================================== #
## Evaluate the base model
median_abs_error, low, high = model_eval(base_model, x_test, y_test)

logger.info(f'Base Model Accuracy:')
logger.info(f'Median Abs Error: {median_abs_error:.2f} % (central 90%: {low:.2f} - {high:.2f} %) ')

save_model(base_model)

price_dicts_general = []
price_dicts_specialized = []
print('\nIndividual Report per coin of interest:\n')
verbose = False
for coin in coins_buy:
    try:
        # Scrape the data of the coin of interest
        x_tr, y_tr, x_test, y_test, df = create_data(coin, n_predictor_days, 
                verbose=verbose, scope=scope, return_single_dim=False)
    except:
        continue

    # Basic Info Logging
    basic_log(coin, logger, df, n_predictor_days, scope)
    
    # Load Base Model
    base_model = load_model()

    # Check the General model
    logger.info('\nGeneral Model')

    # Get Loss
    _ = train_lstm_model(base_model, x_tr, y_tr, x_test, y_test, 
            n_predictor_days, logger=logger, epochs=20, eval_only=True, verbose=False)
    
    # Check historical performance with General Model
    median_abs_error, low, high = model_eval(base_model, x_test, y_test)
    logger.info(f'\nAccuracy:')
    logger.info(f'Error: {median_abs_error:.2f} % ({low:.2f} - {high:.2f} %) ')
    
    # Predict with general model
    price_dict = predict_coins(coin, base_model, n_predictor_days, verbose=True, plotme=plotme, 
                    scope=scope, flatten=False)

    price_dict[coin].append(low)
    price_dict[coin].append(high)
    price_dict[coin].append(median_abs_error)
    price_dicts_general.append(price_dict)

    logger.info('Prediction:')
    log_price(logger, price_dict)

    #=========================#
    logger.info('\nSpecialized Model')

    if x_tr.shape[0] < n_predictor_days+2:
        eval_only = True
    else:
        eval_only = False
    # Retrain Model
    specialized_model = train_lstm_model(base_model, x_tr, y_tr, x_test, y_test, 
                            n_predictor_days, logger=logger, epochs=75, eval_only=eval_only, verbose=False)

    ## Check historical performance with Specialized Model
    median_abs_error, low, high = model_eval(specialized_model, x_test, y_test)
    logger.info(f'\nAccuracy:')
    logger.info(f'Error: {median_abs_error:.2f} % ({low:.2f} - {high:.2f} %) ')


    # Perform prediction for tomorrow based on the last few days:
    price_dict = predict_coins(coin, specialized_model, n_predictor_days, verbose=True, 
                    plotme=plotme, scope=scope, flatten=False)
    price_dict[coin].append(low)
    price_dict[coin].append(high)
    price_dict[coin].append(median_abs_error)
    price_dicts_specialized.append(price_dict)

    logger.info('Prediction:')
    log_price(logger, price_dict)



df_general = price_dicts_to_df(price_dicts_general)
df_specialized = price_dicts_to_df(price_dicts_specialized)

df_to_excel(df_general, 'General', scope)
df_to_excel(df_specialized, 'Special', scope)
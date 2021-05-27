import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from stonks.cryptobot import *

# Settings
# coins_buy = coins_coinbase
coins_buy = ['BTC', 'ETH']

# coins_pretrain = coins_toponehundred
coins_pretrain = ['BTC', 'ETH']

# coins_train = coins_coinbase
coins_train = ['BTC', 'ETH']

verbose = True
plotme = False
n_predictor_days = 30
scope = 'daily'
# n_predictor_days = 15
# scope = 'weekly'


# Logging
logger = get_logger(scope)
logger.info(f'Scope: {scope}')

# Pre-train Base Model with many many coins:
x_tr, x_test, x_tr_normed, x_test_normed, _ = create_data(coins_pretrain, n_predictor_days, 
                verbose=verbose, scope=scope)

base_model = get_ann_model(n_predictor_days=n_predictor_days, verbose=verbose)

base_model = train_ann_model(base_model, x_tr, x_test, x_tr_normed, x_test_normed, n_predictor_days, 
                logger=logger, verbose=verbose)

# Re-train the pretrained model with the Coinbase set
x_tr, x_test, x_tr_normed, x_test_normed, _ = create_data(coins_train, n_predictor_days, 
                verbose=verbose, scope=scope)

base_model = train_ann_model(base_model, x_tr, x_test, x_tr_normed, x_test_normed, n_predictor_days, 
                logger=logger, verbose=verbose)

## Evaluate the base model
_, x_test_tmp, _, y_test_tmp = unpack_x_y(x_tr, x_test, x_tr_normed, x_test_normed, n_predictor_days)
median_abs_error, low, high = model_eval(base_model, x_test_tmp, y_test_tmp)
logger.info(f'Base Model Accuracy:')
logger.info(f'Median Abs Error: {median_abs_error:.2f} % (central 90%: {low:.2f} - {high:.2f} %) ')

save_model(base_model)


print('\nIndividual Report per coin of interest:\n')
verbose = False
for coin in coins_buy:
    try:
        # Scrape the data of the coin of interest
        x_tr, x_test, x_tr_normed, x_test_normed, df = create_data(coin, n_predictor_days, 
                verbose=verbose, scope=scope)
    except:
        continue
    # Basic Info Logging
    basic_log(coin, logger, df, n_predictor_days, scope)
    
    # Specialized_model: The Base Model plus some extra training on the coin of interest:
    specialized_model = load_model()

    # Perform predictions

    # Using the General Model
    logger.info('\nGeneral Prediction')
    specialized_model = train_ann_model(specialized_model, x_tr, x_test, x_tr_normed, x_test_normed, 
                    n_predictor_days, logger=logger, epochs=20, eval_only=True, verbose=False)
    price_dict = predict_coins(coin, specialized_model, n_predictor_days, verbose=True, plotme=plotme, 
                    scope=scope)
    log_price(logger, price_dict)

    # Using the Specialized Model
    logger.info('\nSpecialized Prediction')

    if x_tr_normed.shape[0] < n_predictor_days+2:
        eval_only = True
    else:
        eval_only = False
        
    specialized_model = train_ann_model(specialized_model, x_tr, x_test, x_tr_normed, x_test_normed, 
                            n_predictor_days, logger=logger, epochs=20, eval_only=eval_only, verbose=False)
    
    # Perform prediction for tomorrow based on the last few days:
    price_dict = predict_coins(coin, specialized_model, n_predictor_days, verbose=True, plotme=plotme, scope=scope)
    # Log the info as a report:
    log_price(logger, price_dict)




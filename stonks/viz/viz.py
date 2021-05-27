import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def plot_yields(stock_yields, title=''):
    lossprob = (np.sum(stock_yields < 1)/  len(stock_yields))*100
    percentile_1 = np.percentile(stock_yields*100, 1) -100
    percentile_10 = np.percentile(stock_yields*100, 10) -100
    
    EV = 100*np.nanmedian(stock_yields) - 100

    fig = plt.figure()
    sns.set()
    sns.distplot(stock_yields*100 - 100, hist=True, kde=True)
    plt.xlabel(f'Stock yields [%]')
    plt.plot([100, 100], [0, plt.ylim()[1]], color='grey')
    if title == '':
        plt.title(f'EV: {EV:.1f} %, P(loss): {lossprob:.1f} %, \n1%: {percentile_1:.1f} %, 10%: {percentile_10:.1f} %')
    else:
        plt.title(f'{title}\nEV: {EV:.1f} %, P(loss): {lossprob:.1f} %, \n1%: {percentile_1:.1f} %, 10%: {percentile_10:.1f} %')
    return fig
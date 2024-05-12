from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd

import matplotlib.ticker as mtick
def plot_quintiles(trade_log, column, bins=5, title=None, aggregate_dict={'net P/L %': 'mean'}):
    quintiles = trade_log.groupby(pd.qcut(trade_log[column], bins)).aggregate(aggregate_dict)

    fig, ax = plt.subplots(figsize=(4, 4)) 
    ax.bar(range(1, bins+1), quintiles[next(iter(aggregate_dict))], color='green')
    ax.set_title(title, fontweight ='bold')
    ax.set_xlabel("Quantile (low to high)")
    ax.set(xticks=range(1, bins+1))
    ax.set_ylabel("Avg. net profit/trade")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    plt.show()

def plot_multiple_quintiles(trade_log, column, bins=5, title=None, aggregate_dict={'net P/L %': 'mean'}, figsize=(8, 5)):
    fig, axes = plt.subplots(2, 3, figsize=figsize)

    for ax, trade_log_split in zip(axes.reshape(-1), np.array_split(trade_log, axes.size)):
        quintiles = trade_log_split.groupby(pd.qcut(trade_log_split[column], bins)).aggregate(aggregate_dict)
        
        ax.bar(range(1, bins+1), quintiles[next(iter(aggregate_dict))], color='green')
        start_date = trade_log_split['datetime_in'].iloc[0].strftime('%b %Y')
        ax.set_title(f'>{start_date}')
        ax.set(xticks=range(1, bins+1))
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    
    fig.suptitle(title, fontweight ='bold')
    fig.tight_layout()
    plt.show()
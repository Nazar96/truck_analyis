from sklearn.base import ClusterMixin
from sklearn.linear_model import Ridge

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class TSTrendEstimator(ClusterMixin):
    """
    Get Ridge regression coeficient for each point of time series
    
    Parameters
    ------------
    rolling_window_size: Rolling window size for time series smoothing
    
    n_lags: size of window to estimate Ridge coef
    
    aplha: Ridge alpha
    
    """
    def __init__(self, rolling_window_size=6, n_lags=6, alpha=1, max_coef=1.5, min_coef=-1.5):
        self.rolling_window_size =rolling_window_size
        self.n_lags = n_lags
        self.alpha = alpha
        self.max_coef = max_coef
        self.min_coef = min_coef
    
    def _get_status(self, estimation):
        if estimation >= self.max_coef:
            return 1
        if estimation <= self.min_coef:
            return -1
        return 0
        
    
    
    def fit_predict(self, X):
        X = X.copy()
                
        # Different approach for plural or sinle time series 
        if isinstance(X.index, pd.core.indexes.multi.MultiIndex):
            multiindex = True
            group_index = X.index.get_level_values(0)
        else:
            multiindex = False
            group_index = None
        
        # Rolling window group mean
        if multiindex:
            X = X.groupby(group_index).rolling(self.rolling_window_size).mean()
            X.index = X.index.droplevel(0)
            Y = pd.concat([X.groupby(group_index).shift(i) for i in range(self.n_lags)], axis=1)
        else:
            X = X.rolling(self.rolling_window_size).mean()
            Y = pd.concat([X.shift(i) for i in range(self.n_lags)], axis=1)
        
        mask = (Y.isna().sum(axis=1) == 0).values
        Y = Y.loc[mask].values.T
        
        # Ridge coef        
        x = np.arange(self.n_lags).reshape(-1, 1)
        model = Ridge(alpha=self.alpha).fit(x, Y)

        coef = -1 * model.coef_.flatten()
        res = X.to_frame('X')
        res['estimation'] = np.nan
        res['status'] = np.nan
        res.loc[mask, 'estimation'] = coef
        res.loc[mask, 'status'] = list(map(self._get_status, coef))
        
        return res
    
    
def plot_TSTrendEstimation(df, value='X', coef='estimation', status='status', max_coef=1.5, min_coef=-1.5):
    
    fig, axs = plt.subplots(2,1, sharex=True, figsize=(25,10))

    df[value].plot(color='green', ax=axs[0])
    axs[0].set_ylabel(value)

    df[coef].plot(ax=axs[1])
    axs[1].set_ylabel('coef')

    axs[1].axhline(y=0, color='r', linestyle='-', alpha=0.5)
    axs[1].axhline(y=min_coef, color='r', linestyle='--', alpha=0.25)
    axs[1].axhline(y=max_coef, color='r', linestyle='--', alpha=0.25) 

    plt.show()

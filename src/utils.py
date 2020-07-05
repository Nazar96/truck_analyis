from sklearn.base import ClusterMixin
from sklearn.linear_model import Ridge

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class TSTrendEstimator(ClusterMixin):
    """
    Get Ridge regression coeficient for each point of time series
    
    Parameters
    ------------
    rolling_window_size: Rolling window size for time series smoothing
    
    n_lags: size of window to estimate Ridge coef
    
    aplha: Ridge alpha
    
    """
    def __init__(self, rolling_window_size=6, n_lags=6, alpha=1):
        self.rolling_window_size =rolling_window_size
        self.n_lags = n_lags
        self.alpha = alpha
    
    
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
        res.loc[mask, 'estimation'] = coef
        
        return res
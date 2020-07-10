from sklearn.base import ClusterMixin
from sklearn.linear_model import Ridge

import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



import numpy as np
from matplotlib import pyplot as plt
from plotly import express as px
from datetime import datetime
import seaborn as sns

import shapely
from shapely.geometry import Point
import zipfile
from tqdm import tqdm_notebook as tqdm
from copy import deepcopy, copy
from datetime import timedelta
from collections import OrderedDict
import gc
import re

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from warnings import simplefilter
simplefilter('ignore')


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
    
    def _get_status(self, score):
        if score >= self.max_coef:
            return 1
        if score <= self.min_coef:
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
        res['score'] = np.nan
        res['status'] = np.nan
        res.loc[mask, 'score'] = coef
        res.loc[mask, 'status'] = list(map(self._get_status, coef))
        
        return res
    
    
def plot_TSTrendEstimation(df, value='X', coef='score', status='status', max_coef=1, min_coef=-1, n_lags=6):
   
    fig, axs = plt.subplots(2,1, sharex=True, figsize=(25,10))    

    for status, color in zip([0, -1, 1], ['green', 'blue', 'red']):

        mask = (df['status'] == status)
        
        ids = np.where(mask)[0]
        for i in range(n_lags):
            ids = np.unique(np.concatenate((ids, ids-i)))    
        max_id = mask.shape[0] - 1
        ids[ids < 0] = 0
        ids[ids > max_id] = max_id
        
        tmp = pd.Series(index=df.index)
        tmp.iloc[ids] = df.iloc[ids][value]
        tmp.plot(color=color, ax=axs[0], linewidth=5)
    
    axs[0].set_ylabel(value)

    
    df[coef].plot(ax=axs[1])
    axs[1].set_ylabel('coef')

    axs[1].axhline(y=0, color='r', linestyle='-', alpha=0.5)
    axs[1].axhline(y=min_coef, color='r', linestyle='--', alpha=0.25)
    axs[1].axhline(y=max_coef, color='r', linestyle='--', alpha=0.25) 

    plt.show()


######################################################################


filter_zones = [
    {
        'lon': 52.46469498,
        'lat': 55.71164005,
        'radius': 50_000,
        'name': 'nab_cheln',
    },
]


def value2tuple(value):
    if isinstance(value, (tuple, list)):
        return value
    return (value,)


def init_filter_zones(filter_zones, epsg):
    result = {}
    for zone in filter_zones:
        result.update({
            zone['name']: gpd.GeoSeries([Point(zone['lon'], zone['lat'])],
                                        crs={'init': f'epsg:{epsg}'}).to_crs(epsg=3576). \
                buffer(zone['radius']).iloc[0]
        })
    return result


def dfPolar2gdfCartesian(df, epsg_init=4326):
    df = gpd.GeoDataFrame(df,
                          geometry=gpd.points_from_xy(df['lon'], df['lat']),
                          crs={'init': f'epsg:{epsg_init}'})
    df.to_crs(epsg=3576, inplace=True)
    return df


def prep_df(df):
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', drop=False, inplace=True)
    df.sort_index(inplace=True)
    df['Trip Distance padded'] = df['Trip Distance (km)'].fillna(method='pad').fillna(0)
    return df


class TripLabel():
    def __init__(self, path=None, df_dict=None, split_criterios=('time', 'distance'),
                 split_params={'hours': 12}, **params):
        """
        path: path to zipfile
        df_dict: dictionary of dataframes
        """

        self.trip_diff_method = \
            {
                'time': self.trip_diff_by_time_delta,
                'distance': self.trip_diff_by_trip_distance,
            }

        self.split_criterios = split_criterios
        self.split_params = split_params
        self._params = params

        if df_dict is None:
            setattr(zipfile.ZipFile, 'keys', lambda x: x.namelist())
            self._df_dict = zipfile.ZipFile(path)
        else:
            self._df_dict = df_dict

        self._filenames = self.keys()

    def set_filenames(self, filenames='all'):
        """Set files to label trips"""

        self._filenames = filenames
        if self._filenames == 'all':
            self._filenames = self.keys()
        elif isinstance(filenames, str):
            self._filenames = [self._filenames]
        return self

    def get_filenames(self):
        return self._filenames

    def __iter__(self):
        return self.gen_labeled_trips()

    def __len__(self):
        return len(self.get_filenames())

    def keys(self):
        """Get all file names in zip"""
        return list(self._df_dict.keys())

    def gen_labeled_trips(self):
        """Generated dataframes with labeled trips"""

        trip_id = [0]
        for df in self._gen_data():
            trip_id = self.label(df, self.split_criterios) + trip_id[-1]
            df['trip_id'] = trip_id
            yield df

    def _gen_data(self):
        """Generate original dataframe from zipfile"""
        for filename in self._filenames:
            yield self.read_file(filename)

    def read_file(self, filename):
        """Read and prepare raw data from zipfile"""

        if isinstance(self._df_dict, dict):
            df = self._df_dict[filename]
        else:
            df = pd.read_csv(self._df_dict.open(filename), **self._params)
        df = prep_df(df)
        return df

    def label(self, df, criterios):
        """Return list of trip labels for dataframe"""
        trip_diff_mask = [self.trip_diff_method[criterio](df, self.split_params) for criterio in criterios]
        trip_diff_mask = (np.sum(trip_diff_mask, axis=0) > 0).tolist()
        trip_id = self._label_trip(trip_diff_mask)
        return trip_id

    @staticmethod
    def trip_diff_by_trip_distance(df, params):
        """Return list of trip dist diffs for dataframe"""
        trip_diff_mask = np.asarray(df['Trip Distance padded'].diff().fillna(-1) < 0)
        return trip_diff_mask

    @staticmethod
    def trip_diff_by_time_delta(df, params):
        """Return list of trip time diffs for dataframe"""
        delta_t = timedelta(**params)
        trip_diff_mask = np.asarray(df['time'].diff().fillna(delta_t) >= delta_t)
        return trip_diff_mask

    @staticmethod
    def _label_trip(trip_diff_mask):
        """Convert binary mask of trip diffs to trip_id"""

        trip_id = deepcopy(trip_diff_mask)
        flag = True
        trip_counter = 0

        for i in range(1, len(trip_id)):
            if (trip_id[i] == False):
                flag = False

            if (trip_id[i] == True) and (flag == True):
                trip_id[i] = False

            if (trip_id[i] == True):
                flag = True

        for i in range(len(trip_id)):
            if trip_id[i] == True:
                trip_counter += 1
            trip_id[i] = trip_counter

        return np.asarray(trip_id)


class TripFilter():
    def __init__(self, df, filter_zones=filter_zones, epsg_init=4326):
        self.epsg_init = epsg_init
        self._filter_zones = init_filter_zones(filter_zones, self.epsg_init)

        self._df = df
        self._invalid_trip_set = set()
        self._filters = \
            OrderedDict({
                'distance': self.filter_by_total_distance,
                'jumping': self.filter_by_jumping,
                'zone': self.filter_by_zone
            })

    def filter_by(self, criterios, verbose=True):
        criterios = value2tuple(criterios)
        for criterio in self._filters.keys():
            if criterio in criterios:
                self._filters[criterio]().remove_invalid_trips()
        return self

    def get_dataframe(self):
        return self._df

    @property
    def dataframe(self):
        return self.get_dataframe()

    def get_invalid_trip_set(self):
        return copy(self._invalid_trip_set)

    def update_invalid_trips_set(self, invalid_list):
        self._invalid_trip_set = self._invalid_trip_set.union(invalid_list)
        return self

    def label_by_filter_zones(self, gdf):
        for zone_name in self._filter_zones.keys():
            lambda_filter = lambda p: self._filter_zones[zone_name].contains(p)
            gdf[zone_name] = gdf['geometry'].apply(lambda_filter)
        return gdf

    def conv2gpd(self, df, freq, columns=[]):
        gdf = []
        for trip_id in df.trip_id.unique():
            tmp = df.query('trip_id == @trip_id')[set(['lat', 'lon'] + columns)]
            tmp = tmp.resample(freq).mean().dropna()
            tmp['trip_id'] = trip_id
            gdf.append(tmp)
        gdf = dfPolar2gdfCartesian(pd.concat(gdf))
        return gdf

    def out_of_filter_zones_counter(self, freq='5T'):
        zone_names = list(self._filter_zones.keys())
        gdf = self.conv2gpd(self._df, freq=freq)
        gdf = self.label_by_filter_zones(gdf)

        out_of_filter_zone_df = ~gdf[zone_names]
        out_zone_counter_df = out_of_filter_zone_df.groupby(gdf['trip_id'])[zone_names]
        out_zone_counter_df = out_zone_counter_df.sum().astype(int)
        return out_zone_counter_df

    def filter_by_zone(self, thresh=0, freq='5T'):
        out_zone_counter_df = self.out_of_filter_zones_counter(freq=freq)
        out_zone_counter_df = out_zone_counter_df.min(axis=1).astype(int)
        out_zone_trips_list = out_zone_counter_df[out_zone_counter_df <= thresh].index.tolist()
        self.update_invalid_trips_set(out_zone_trips_list)
        return self

    def filter_by_total_distance(self, thresh=500):
        total_trip_dist_df = self._df.groupby('trip_id')['Trip Distance (km)'].agg(['min', 'max'])
        total_trip_dist_df = (total_trip_dist_df['max'] - total_trip_dist_df['min']).fillna(0)
        short_dist_trips_list = total_trip_dist_df[total_trip_dist_df <= thresh].index.tolist()
        self.update_invalid_trips_set(short_dist_trips_list)
        return self

    def filter_by_jumping(self, thresh=10, count_thresh=1):
        jumping_trip_list = []
        for trip_id in self._df['trip_id'].unique():
            df = self._df.query('trip_id == @trip_id')
            value = (df['Trip Distance padded'].diff() >= thresh).values.sum()
            if value >= count_thresh:
                jumping_trip_list.append(trip_id)
        self.update_invalid_trips_set(jumping_trip_list)
        return self

    def remove_invalid_trips(self):
        self._df = self._df[~self._df['trip_id'].isin(self._invalid_trip_set)]
        self._invalid_trip_set = set()
        return self
# Тут читаю данные, подготоавливаю и съедаю

import pandas as pd
import numpy as np
import sys
sys.path.append('../')
from src.utils import spline_score_multiple_cars, score2cat


def main(argv):
    
    result_path = argv[1]
    
    data_path = '/home/jovyan/remote_shared_data/dsdiag222/temporary/from_ntc_batching/2020_04_26_2020_06_30_valid_distance_upd.csv'
    usecols = ['time', 'speed', 'car_vin']

    # READ DATA
    df = pd.read_csv(data_path, usecols=usecols)
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)

    # RESAMPLE DATA
    resample_freq = '10s'
    df_resampled = df.groupby('car_vin')['speed'].resample(resample_freq).mean()
    df_resampled.replace(0, np.nan, inplace=True)

    # SEGMENTATION
    score = spline_score_multiple_cars(df_resampled)
    score['score'] = score2cat(score['score'])
    
    score.to_csv(result_path)
    
if __name__ == '__main__':
    main(sys.argv)
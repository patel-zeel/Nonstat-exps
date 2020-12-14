import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from polire.interpolate import Kriging
import psutil
import multiprocessing as mp
from time import time
import pickle
import warnings
warnings.filterwarnings('ignore')
import logging
logging.disable()

main_path = '/home/patel_zeel/Nonstat-exps/gp_extra/'
df = pd.read_csv(main_path+'data/beijing_AQI.csv').rename(columns={'PM25_Concentration':'PM25','longitude':'long','latitude':'lat'})
df = df.set_index('time').sort_index()
#print('unique timestamps are',len(df.index.unique()))
useful_ts = []
for ts in df.index.unique():
  if(len(df.loc[ts])==36):
    useful_ts.append(ts)
df = df.loc[useful_ts]
df['PM25'] = df['PM25'].astype(float)
#print('unique timestamps after removing missing entry time-stamps are',len(useful_ts))
df.columns

K = 3 #  Number of folds
n_val = 6 # Number of validation stations

splitter = KFold(K, shuffle=True, random_state=0)
stations = np.sort(df['station_id'].unique())
folds={i:{'train':None,'val':None,'test':None} for i in range(K)}
for i, (train_val, test) in enumerate(splitter.split(stations)):
    folds[i]['train'] = stations[train_val[:-n_val]]
    folds[i]['val'] = stations[train_val[-n_val:]]
    folds[i]['test'] = stations[test]
    
###########################
# Data preperation
###########################
data = {i:{'train_Xy':None,'val_Xy':None,'test_Xy':None} for i in range(K)}
for fold in range(K):
    for part in ['train','val','test']:
        data[fold][part+'_Xy'] = (df[df.station_id.isin(folds[fold][part])][['long', 'lat']], 
                                  df[df.station_id.isin(folds[fold][part])][['PM25']])
recompute = True
path = main_path+'results/raw_kriging/'
ts_n = int(sys.argv[1])
ts = df.index.unique()[ts_n]
fold = int(sys.argv[2])
result_dict = {'best_model':None,
           'best_val_error':np.inf,
           'best_hyperpara':{'nlags':None},
           'train_Xy':None,
           'test_Xy':None,
           'val_Xy':None,
           'pred_y':None,
           'test_y':None,
           'RMSE':None}
#print(os.path.exists(path+'ts_'+str(ts)+'_fold_'+str(fold)))
if os.path.exists(path+'ts_'+str(ts)+'_fold_'+str(fold)):
    if not recompute:
        sys.exit()
        
scaler = StandardScaler()
for nlag in [3,4,5,6,7,8,9]: # 6 is default
#     print('checking',nlag)
    model = Kriging(nlags=nlag)
    model.fit(scaler.fit_transform(data[fold]['train_Xy'][0].loc[ts].values), data[fold]['train_Xy'][1].loc[ts].values)
    pred_y = model.predict(scaler.transform(data[fold]['val_Xy'][0].loc[ts].values))
    error = mean_squared_error(data[fold]['val_Xy'][1].loc[ts].values, pred_y, squared=True)
    if error<result_dict['best_val_error']:
        result_dict['best_model'] = model
        result_dict['best_hyperpara']['nlags'] = nlag
        result_dict['best_val_error'] = error

model = result_dict['best_model']
pred_y = model.predict(scaler.transform(data[fold]['test_Xy'][0].loc[ts].values))
for part in ['train','val','test']:
    result_dict[part+'_Xy'] = (data[fold][part+'_Xy'][0].loc[ts].values, data[fold][part+'_Xy'][1].loc[ts].values)
result_dict['pred_y'] = pred_y
result_dict['test_y'] = data[fold]['test_Xy'][1].loc[ts].values
# print(mean_squared_error(result_dict['test_y'], pred_y, squared=False))
# print('test_y', result_dict['test_y'].squeeze().astype(int).tolist())
# print('pred_y', pred_y.astype(int).tolist())
pd.to_pickle(result_dict, path+'ts_'+str(ts)+'_fold_'+str(fold))
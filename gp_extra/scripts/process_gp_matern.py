import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
import psutil
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from time import time

main_path = '/home/patel_zeel/Nonstat-exps/gp_extra/'
df = pd.read_csv(main_path+'data/beijing_AQI.csv').rename(columns={'PM25_Concentration':'PM25','longitude':'long','latitude':'lat'})
df = df.set_index('time').sort_index()
# print('unique timestamps are',len(df.index.unique()))
useful_ts = []
for ts in df.index.unique():
  if(len(df.loc[ts])==36):
    useful_ts.append(ts)
df = df.loc[useful_ts]
df['PM25'] = df['PM25'].astype(float)
# print('unique timestamps after removing missing entry time-stamps are',len(useful_ts))

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
        
from GPy.models import GPRegression as GPR
from GPy.kern import Matern32

result_dict = {'best_model':None,
           'best_val_error':np.inf,
           'best_hyperpara':{'ls_init':None},
           'train_Xy':None,
           'test_Xy':None,
           'val_Xy':None,
           'pred_y':None,
           'test_y':None,
           'RMSE':None}
recompute = True
path = main_path+'results/raw_gp_matern/'
ts_n = int(sys.argv[1])
ts = df.index.unique()[ts_n]
fold = int(sys.argv[2])
if os.path.exists(path+'ts_'+str(ts)+'_fold_'+str(fold)):
    if not recompute:
        sys.exit()
scaler = StandardScaler()
for ls_init in [0.01,0.1,1.,10.,100.]: # 1 is default
    model = GPR(scaler.fit_transform(data[fold]['train_Xy'][0].loc[ts].values), 
                data[fold]['train_Xy'][1].loc[ts].values,
               Matern32(input_dim=2, active_dims=[0,1], ARD=True))
    model.kern.lengthscale = [ls_init, ls_init]
    model.optimize()
    pred_y, _ = model.predict(scaler.transform(data[fold]['val_Xy'][0].loc[ts].values))
    error = mean_squared_error(data[fold]['val_Xy'][1].loc[ts].values, pred_y, squared=False)
    if error<result_dict['best_val_error']:
        result_dict['best_model'] = model
        result_dict['best_hyperpara']['ls_init'] = ls_init
        result_dict['best_val_error'] = error

model = result_dict['best_model']
pred_y, _ = model.predict(scaler.transform(data[fold]['test_Xy'][0].loc[ts].values))
for part in ['train','val','test']:
    result_dict[part+'_Xy'] = (data[fold][part+'_Xy'][0].loc[ts].values, data[fold][part+'_Xy'][1].loc[ts].values)
result_dict['pred_y'] = pred_y
result_dict['test_y'] = data[fold]['test_Xy'][1].loc[ts].values
model_path = path+'model_ts_'+str(ts)+'_fold_'+str(fold)
if os.path.exists(model_path):
    os.remove(model_path)
result_dict['best_model'] = None
print(mean_squared_error(result_dict['test_y'], pred_y, squared=False), result_dict['best_hyperpara']['N'])
print('test_y', result_dict['test_y'].squeeze().astype(int).tolist())
print('pred_y', pred_y.squeeze().astype(int).tolist())
print('len', scaler.transform(data[fold]['test_Xy'][0].loc[ts].values))
pd.to_pickle(model, main_path+'scripts/scratch/test_model_matern')
# pd.to_pickle(result_dict, path+'ts_'+str(ts)+'_fold_'+str(fold))
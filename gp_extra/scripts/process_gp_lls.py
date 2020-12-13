import pandas as pd
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import sys
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from polire.interpolate import Kriging
from gpflow.transforms import Logistic
import psutil
import multiprocessing as mp
from time import time

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

n_ts = 24 # Number of timestamps
K = 4 #  Number of folds
n_val = 2 # Number of validation stations

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
        
import tensorflow as tf
import gpflow
from gpflow.models import GPR
from gpflow.kernels import LLS, RBF, Matern32
from gpflow.training.scipy_optimizer import ScipyOptimizer
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
config = tf.ConfigProto(intra_op_parallelism_threads=8, 
                        inter_op_parallelism_threads=32, 
                        allow_soft_placement=True,
                        gpu_options=gpu_options,
                        device_count = {'CPU': psutil.cpu_count()})

with tf.Session(config=config) as sess:
    result_dict = {'best_model':None,
               'best_val_error':np.inf,
               'best_hyperpara':{'N':None},
               'train_Xy':None,
               'test_Xy':None,
               'val_Xy':None,
               'pred_y':None,
               'test_y':None,
               'RMSE':None}
    recompute = False
    path = main_path+'results/raw_gp_lls/'
    ts_n = int(sys.argv[1]) # Number of time-stamps to process
    ts = df.index.unique()[ts_n]
    n_ts = 24
    fold = int(sys.argv[2])
    K = 4
    if os.path.exists(path+'ts_'+str(ts)+'_fold_'+str(fold)):
        if not recompute:
            sys.exit()
    print('chk2')
    scaler = StandardScaler()
    for N in [3,4,5,6,7]:
        opt = ScipyOptimizer()
        model = GPR(scaler.fit_transform(data[fold]['train_Xy'][0].loc[ts].values), 
                    data[fold]['train_Xy'][1].loc[ts].values,
                   LLS(2, scaler.fit_transform(data[fold]['train_Xy'][0].loc[ts].values), N, active_dims=[0,1]))
        model.clear()
        model.kern.l_bar.transform = Logistic(10**-3, 10**3)
        model.kern.l_lengthscales.transform = Logistic(10**-3, 10**3)
        model.kern.variance.transform = Logistic(10**-3, 10**3)
        model.compile()
        opt.minimize(model)
        pred_y, _ = model.predict_y(scaler.transform(data[fold]['val_Xy'][0].loc[ts].values))
        error = mean_squared_error(data[fold]['val_Xy'][1].loc[ts].values, pred_y, squared=True)
        if error<result_dict['best_val_error']:
            result_dict['best_model'] = model
            result_dict['best_hyperpara']['N'] = N
            result_dict['best_val_error'] = error
    model = result_dict['best_model']
    pred_y, _ = model.predict_y(scaler.transform(data[fold]['test_Xy'][0].loc[ts].values))
    for part in ['train','val','test']:
        result_dict[part+'_Xy'] = (data[fold][part+'_Xy'][0].loc[ts].values, data[fold][part+'_Xy'][1].loc[ts].values)
    result_dict['pred_y'] = pred_y
    result_dict['test_y'] = data[fold]['test_Xy'][1].loc[ts].values
    model_path = path+'model_ts_'+str(ts)+'_fold_'+str(fold)
    if os.path.exists(model_path):
        os.remove(model_path)
    gpflow.saver.Saver().save(model_path, model)
    result_dict['best_model'] = None
    pd.to_pickle(result_dict, path+'ts_'+str(ts)+'_fold_'+str(fold))
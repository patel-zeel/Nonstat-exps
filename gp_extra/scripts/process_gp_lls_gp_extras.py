import os
import sys
import pandas as pd
import numpy as np
from time import time
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, GridSearchCV
from scipy.optimize import differential_evolution
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels \
    import ConstantKernel as C, Matern
from sklearn.metrics import mean_squared_error
from gp_extras.kernels import LocalLengthScalesKernel
from sklearn.metrics import mean_squared_error
import scipy
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')
import multiprocessing as mp
## Terminology is consistent with paper 
# Paper name: Nonstationary Gaussian Process Regression Using Point Estimates of Local Smoothness

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
kern_name = 'matern'
path = main_path+'results/raw_gp_lls_'+kern_name+'_gp_extras/'
ts_n = int(sys.argv[1])
ts = df.index.unique()[ts_n]
fold = int(sys.argv[2])
if os.path.exists(path+'ts_'+str(ts)+'_fold_'+str(fold)):
    if not recompute:
        sys.exit()
        
def de_optimizer(obj_func, initial_theta, bounds):
    res = differential_evolution(lambda x: obj_func(x, eval_gradient=False),
                                 bounds, disp=False, polish=False, seed=0)
    return res.x, obj_func(res.x, eval_gradient=False)
scaler = StandardScaler()

for N in [3,4,5,6,7,8,9]:
    print('checking', N)
    kernel_lls = C(1.0, (10**-5, 10**5)) \
    * LocalLengthScalesKernel.construct(scaler.fit_transform(data[fold]['train_Xy'][0].loc[ts].values), l_samples=N)
    model = GaussianProcessRegressor(kernel=kernel_lls, optimizer=de_optimizer, random_state=0)
    model.fit(scaler.fit_transform(data[fold]['train_Xy'][0].loc[ts].values), data[fold]['train_Xy'][1].loc[ts].values)
    pred_y = model.predict(scaler.transform(data[fold]['val_Xy'][0].loc[ts].values))
    error = mean_squared_error(data[fold]['val_Xy'][1].loc[ts].values, pred_y, squared=False)
    if error<result_dict['best_val_error']:
        result_dict['best_model'] = model
        result_dict['best_hyperpara']['N'] = N
        result_dict['best_val_error'] = error

model = result_dict['best_model']
pred_y = model.predict(scaler.transform(data[fold]['test_Xy'][0].loc[ts].values))
for part in ['train','val','test']:
    result_dict[part+'_Xy'] = (data[fold][part+'_Xy'][0].loc[ts].values, data[fold][part+'_Xy'][1].loc[ts].values)
result_dict['pred_y'] = pred_y
result_dict['test_y'] = data[fold]['test_Xy'][1].loc[ts].values
pd.to_pickle(result_dict, path+'ts_'+str(ts)+'_fold_'+str(fold))
print(mean_squared_error(data[fold]['test_Xy'][1].loc[ts].values, pred_y, squared=False))
counter = pd.read_pickle(path+'_count')
counter += 1
pd.to_pickle(counter, path+'_count')
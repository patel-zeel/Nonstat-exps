import os
import sys
import pandas as pd
import numpy as np
from time import time
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import scipy
from sklearn.cluster import KMeans
from sklearn.covariance import EmpiricalCovariance as EC
import warnings
warnings.filterwarnings('ignore')

from NSGPy.NumPy import LLS

main_path = '/home/patel_zeel/Nonstat-exps/gp_extra/'
df = pd.read_csv(main_path+'data/beijing_AQI.csv').rename(columns={'PM25_Concentration':'PM25','longitude':'long','latitude':'lat'})
df = df.set_index('time').sort_values(['time', 'station_id'])
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

recompute = True
path = main_path+'results/raw_gp_lls_self_greedy/'
ts_n = int(sys.argv[1])
ts = df.index.unique()[ts_n]
fold = int(sys.argv[2])
g_ts = int(sys.argv[4])
kernel = sys.argv[3]
if os.path.exists(path+kernel+'/ts_'+str(ts)+'_fold_'+str(fold)):
    if not recompute:
        sys.exit()

scaler = StandardScaler()
g_ts_o = df.index.unique()[:g_ts]
tmp_data = data[fold]['train_Xy'][1].loc[g_ts_o].values.reshape(-1, g_ts, order='F').T
emp_cov = EC().fit(tmp_data).covariance_
# print(emp_cov.shape)
A = []
A_bar = list(range(emp_cov.shape[0]))
for each in range(9):
    if len(A)==0:
        scores_num = emp_cov.diagonal()
    else:
        scores_num = emp_cov.diagonal() - ((emp_cov[:, A]@np.linalg.pinv(emp_cov[np.ix_(A, A)]))*emp_cov[:, A]).sum(axis=1)
    if len(A_bar)==0:
        scores_din = emp_cov.diagonal()
    else:
#         print(each, emp_cov.shape, emp_cov[np.ix_(A_bar, A_bar)].shape)
        scores_din = emp_cov.diagonal() - ((emp_cov[:, A_bar]@np.linalg.pinv(emp_cov[np.ix_(A_bar, A_bar)]))*emp_cov[:, A_bar]).sum(axis=1)
    delta = scores_num / scores_din
    delta[A] = -np.inf
    point = np.argmax(delta)
#     print(A)
#     print(A_bar)
#     print(point)
    A.append(point)
    A_bar.remove(point)

for N in [3,4,5,6,7,8,9]:
    model = LLS(2,N_l_bar=N,kernel=kernel, bounds=(10**-3, 10**5), l_isotropic=False, N_l_bar_method='greedy')
    model.fit(scaler.fit_transform(data[fold]['train_Xy'][0].loc[ts].values), data[fold]['train_Xy'][1].loc[ts].values, A[:N])
    pred_y = model.predict(scaler.transform(data[fold]['val_Xy'][0].loc[ts].values), False)
    error = mean_squared_error(data[fold]['val_Xy'][1].loc[ts].values, pred_y, squared=False)
    if error<result_dict['best_val_error']:
        result_dict['best_model'] = model
        result_dict['best_hyperpara']['N'] = N
        result_dict['best_val_error'] = error

model = result_dict['best_model']
pred_y = model.predict(scaler.transform(data[fold]['test_Xy'][0].loc[ts].values), False)
for part in ['train','val','test']:
    result_dict[part+'_Xy'] = (data[fold][part+'_Xy'][0].loc[ts].values, data[fold][part+'_Xy'][1].loc[ts].values)
result_dict['pred_y'] = pred_y
result_dict['test_y'] = data[fold]['test_Xy'][1].loc[ts].values
print(mean_squared_error(result_dict['test_y'], pred_y, squared=False), result_dict['best_hyperpara']['N'])
print('test_y', result_dict['test_y'].squeeze().astype(int).tolist())
print('pred_y', pred_y.squeeze().astype(int).tolist())
print('len', scaler.transform(data[fold]['test_Xy'][0].loc[ts].values))
print(model.params)
pd.to_pickle(model, main_path+'scripts/scratch/test_model')
# print(result_dict)
pd.to_pickle(result_dict, path+kernel+'/ts_'+str(ts)+'_fold_'+str(fold))
# print('saved', path+kernel+'/ts_'+str(ts)+'_fold_'+str(fold))
# counter = pd.read_pickle(path+kernel+'_count')
# counter += 1
# pd.to_pickle(counter, path+kernel+'_count')
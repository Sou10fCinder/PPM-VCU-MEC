from matplotlib import axis
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, ensemble
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, mean_squared_log_error, median_absolute_error, mean_poisson_deviance, mean_gamma_deviance, mean_tweedie_deviance
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
#import seaborn as sns
import xgboost as xgb
from sklearn.metrics import accuracy_score
import argparse
#import shap
from datetime import datetime, date, timedelta
from scipy.optimize import differential_evolution
import pathlib


from typing import Tuple
from sklearn.metrics import r2_score


def r2score(predt: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[str, float]:
    y = dtrain.get_label()
    #print(y)
    return 'R2', float(r2_score(y, predt))

def parameter_tuning_objective_func(X, *args):

    (test_set_ratio, keys, feature_names, df_X, df_Y) = args

    n_estimators = int(X[keys.index('n_estimators')])
    min_child_weight = int(X[keys.index('min_child_weight')])
    eta = X[keys.index('eta')]
    colsample_bytree = X[keys.index('colsample_bytree')]
    max_depth = int(X[keys.index('max_depth')])
    subsample = X[keys.index('subsample')]
    lambda_ = X[keys.index('lambda')]
    learning_rate = X[keys.index('learning_rate')]
    early_stopping_rounds = int(X[keys.index('early_stopping_rounds')])
    n_important_features = int(X[keys.index('n_important_features')])

    args_list = [n_estimators, min_child_weight, eta, colsample_bytree, max_depth, subsample, lambda_, learning_rate, early_stopping_rounds, n_important_features]

    X = df_X.copy()
    y = df_Y.copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_set_ratio
    )
    # performing preprocessing part
    sc = StandardScaler()

    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)


    params = {
        'n_estimators': n_estimators, #500,
        'min_child_weight': min_child_weight, #1, 
        'eta': eta, #0.1, 
        'colsample_bytree': colsample_bytree, #0.9, 
        'max_depth': max_depth, #4,
        'subsample': subsample, #0.9, 
        'lambda': lambda_, #1.0, 
        #'nthread': -1, 
        #'booster' : 'gbtree',
        #'eval_metric': 'rmse', 
        'objective': 'reg:squarederror',
        'learning_rate': learning_rate, #0.01,
        #'verbosity': 2,
        'predictor': 'cpu_predictor',
        'verbosity': 0,
    }

    
    #data_dmatrix = xgb.DMatrix(data=X_train,label=y_train, feature_names=feature_names)
    #xgb_cv = xgb.cv(dtrain=data_dmatrix, params=params, num_boost_round = n_estimators, verbose_eval  = False, nfold=10, metrics = 'rmse')#metrics = 'r2') 

    dtrain = xgb.DMatrix(data=X_train,label=y_train, feature_names=feature_names)
    dtest = xgb.DMatrix(data=X_test,label=y_test, feature_names=feature_names)

    evals_result = {}

    bst = xgb.train(params,
        dtrain,
        num_boost_round=n_estimators,
        evals = [(dtest,'eval'), (dtrain,'train')],
        early_stopping_rounds=early_stopping_rounds,
        evals_result=evals_result,
        verbose_eval=False)

    feature_importances = np.array(list(bst.get_score(importance_type='weight').values()))
    sorted_idx = np.argsort(feature_importances)[::-1][:len(feature_importances)]
    feature_names = np.array(list(bst.get_score(importance_type='weight').keys()))[sorted_idx][:n_important_features]

    

    df_X = df_X.loc[:, feature_names]
    X = df_X.copy()
    y = df_Y.copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_set_ratio
    )
    

    sc = StandardScaler()

    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    data_dmatrix = xgb.DMatrix(data=X_train,label=y_train, feature_names=feature_names)
    xgb_cv = xgb.cv(dtrain=data_dmatrix, params=params, num_boost_round = n_estimators, verbose_eval  = False, early_stopping_rounds=early_stopping_rounds, nfold=20, metrics = 'rmse')#metrics = 'r2') 


    # dtrain = xgb.DMatrix(data=X_train,label=y_train, feature_names=feature_names)
    # dtest = xgb.DMatrix(data=X_test,label=y_test, feature_names=feature_names)


    # bst = xgb.train(params,
    #     dtrain,
    #     num_boost_round=n_estimators,
    #     early_stopping_rounds=early_stopping_rounds,
    #     evals = [(dtest,'eval'), (dtrain,'train')],
    #     verbose_eval=False)

    args_str = dict(zip(keys, args_list))
    
    #score = np.sqrt(mean_squared_error(dtest.get_label(), bst.predict(dtest)))
    score = xgb_cv[-1:].values[0][2]

    print(args_str, score)
    #print("The root mean squared error (RMSE) on test set: {:.4f}".format(score))
    # print("The mean squared error (MSE) on test set: {:.4f}".format(mean_squared_error(dtest.get_label(), bst.predict(dtest))))
    # print("The mean absolute error (MAE) on test set: {:.4f}".format(mean_absolute_error(dtest.get_label(), bst.predict(dtest))))
    # print("The mean absolute percentage error (MAPE) on test set: {:.4f}".format(mean_absolute_percentage_error(dtest.get_label(), bst.predict(dtest))))
    # print("The R square (R2) on test set: {:.4f}".format(np.sqrt(r2_score(dtest.get_label(), bst.predict(dtest)))))
    # print("The median_absolute_error (MEAE) on test set: {:.4f}".format(median_absolute_error(dtest.get_label(), bst.predict(dtest))))

    return score

def dataset_preprocess():
    TEST_SET_RATIO = 0.5

    dir = './'
    filename = 'DataSet_IO_Intensive'
    #filename = 'Power_Prediction_DataSet_GPU_Intensive_hashcat'
    #filename = 'Power_Prediction_DataSet_GPU_Intensive_END_BACKUP'
    dataset = pd.read_excel(dir + filename + '.xlsx')
    size1 = dataset.shape[0]
    #dataset = dataset[(dataset['Used_Time(s)'] <= 5.0) & (dataset['Used_Time(s)'] >= 1.0)]
    #dataset = dataset[(dataset['Used_Time(s)'] >= 1.0)]
    #Net_Active_Power(W)
    #dataset = dataset[~(dataset['Used_Time(s)'] >= 1000.0)]
    #dataset = dataset[(dataset['Container_GPU_Utilization(%)'] > 0) & (dataset['Container_GPU_Utilization(%)'] < 50)]
    #dataset = dataset[(dataset['Container_GPU_Utilization(%)'] > 0)]
    #dataset = dataset[(dataset['Net_Active_Power(W)'] > 0)]
    size2 = dataset.shape[0]

    print(size2)
    print('remove n samples: ', size1 - size2)

    drop_list = ['task_name', 'Active_Power(W)', 'Activte_Energy(J)', 'Task_Energy_Concumption_Total(J)', 'Used_Time(s)', 'Net_Active_Power(W)', 'Frequency(Hz)',
        'Net_Active_Energy(J)', 'ABS_Net_Active_Power(W)', 'ABS_Net_Active_Energy(J)', 'TimeStamps',  'Voltage(V)',
        'Reactive_Energy(J)', 'Annual_Electricity_Consumption(KW/h)', 'Current(A)', 'Power_Factor(PF)', 'Node_Load15', 'Node_Load1',
        #'Container_CPU_Request(%)', #'Container_Memory_Request(MB)', #'Container_GPU_Request', #'Container_IOPS_Request', #'program_name'
        #'file_space_usage(K)', #'Container_GPU_Framebuffer_Mem_Free(MB)',  'Container_GPU_Power_Total', 'Container_GPU_Total_Energy_Consumption(J)',
        #'Container_GPU_Power_Usage(%)',
    ]


    useless_info_list = [
        'inst_retired.any', 'offcore_requests.all_data_rd', 'offcore_requests.demand_data_rd' , 'l2_rqsts.swpf_miss', 'l2_rqsts.all_demand_miss','inst_retired.any#0.40L1MPKI',
        'mem_load_retired.l1_miss' , 'inst_retired.any#0.06L2MPKI', 'mem_load_retired.l2_miss' , 'inst_retired.any#0.02L3MPKI', 'mem_load_retired.l3_miss', 'itlb_misses.walk_pending',
        'dtlb_store_misses.walk_pending', 'cpu_clk_unhalted.distributed', 'dtlb_load_misses.walk_pending', 'br_misp_retired.all_branches', 'inst_retired.any#IpTB1', 'inst_retired.any#IpTB2',
        'br_inst_retired.near_taken', 'inst_retired.any#IpBranch', 'br_inst_retired.all_branches', 'inst_retired.any#IpCall', 'br_inst_retired.near_call', 'br_inst_retired.near_taken.1',
        'br_inst_retired.all_branches.1', 'inst_retired.any#IpFarBranch','br_inst_retired.far_branch:u',
    ]

    extra_info_list = [
        'L1-dcache-loads','L1-dcache-load-misses','LLC-loads',
        'LLC-load-misses','L1-icache-loads','L1-icache-load-misses','dTLB-loads','dTLB-load-misses','iTLB-loads','iTLB-load-misses','L1-dcache-prefetches','L1-dcache-prefetch-misses',
    ]

    node_drop_list = [
        #'Node_Avg_Cpu_Frequency_Max_Hertz(GHz)','Node_Avg_Cpu_Frequency_Min_Hertz(GHz)','Node_Avg_Cpu_Guest_Seconds_Total_Mode_Nice(s)','Node_Avg_Cpu_Guest_Seconds_Total_Mode_User(s)',
        #'Node_Cpu_Seconds_Total_Mode_Iowait(s)','Node_Cpu_Core_Throttles_Total', 'Node_Cpu_Seconds_Total_Mode_Nice(s)','Node_Cpu_Seconds_Total_Mode_Softirq(s)','Node_Cpu_Seconds_Total_Mode_Steal(s)', 
        #Node_Cpu_Seconds_Total_Mode_Irq(s) #Node_Cpu_Seconds_Total_Mode_System(s) #Node_Cpu_Seconds_Total_Mode_User(s) #Node_Cpu_Seconds_Total_Mode_Idel(s)
        'Node_Disk_Io_Now', 'Node_Disk_Read_Bytes_Total(MB)', 'Node_Disk_Written_Bytes_Total(MB)','Node_Filefd_Allocated','Node_Filesystem_Files(M)','Node_Filesystem_Files_Free(M)',
        #Node_Intr_Total(K), 'Container_Fs_Writes_Merged_Total', 'Container_Fs_Writes_Total', 'Container_Fs_Reads_Merged_Total',  'Container_Fs_Reads_Total',
        #'Node_Load1','Node_Load5','Node_Load15','Node_Pressure_Cpu_Waiting_Seconds_Total(s)', 'Node_Pressure_Memory_Stalled_Seconds_Total(s)','Node_Procs_Blocked',
        #'Node_Procs_Running','Node_Schedstat_Running_Seconds_Total(s)','Node_Schedstat_Timeslices_Total(K)','Node_Schedstat_Waiting_Seconds_Total(h)','Mode_Memory_MemTotal_Bytes(GB)',
        #'Node_Memory_Cached_Bytes(GB)','Node_Memory_Buffers_Bytes(GB)','Node_Memory_MemFree_Bytes(GB)','Node_Memory_SwapTotal_Bytes(GB)','Node_Memory_SwapFree_Bytes(GB)',
        'Node_Schedstat_Timeslices_Total(K)', 'Node_Schedstat_Waiting_Seconds_Total(h)','Container_Fs_Io_Current',  'Container_Fs_Sector_Reads_Total',
        'Container_Fs_Sector_Writes_Total', 'Node_Memory_Buffers_Bytes(GB)', 'Mode_Memory_MemTotal_Bytes(GB)', 
        #'Container_GPU_Temperature(C)','Container_GPU_Framebuffer_Mem _Used(MB)','Container_GPU_Framebuffer_Mem_Free(MB)', 'Container_GPU_Utilization(%)',
        #'Container_GPU_SM_Clocks(MHz)', 'Container_GPU_Memory_Clock(MHZ)', 'Container_GPU_DCGM_FI_DEV_MEM_COPY_UTIL', 
        'Node_Memory_MemFree_Bytes(GB)', 'Node_Memory_SwapFree_Bytes(GB)', 'Node_Memory_SwapTotal_Bytes(GB)', 'Node_Memory_Cached_Bytes(GB)',
        'Container_Spec_Memory_Swap_Limit_Bytes(B)', 'Process_Virtual_Memory_Bytes(GB)', 'Process_Resident_Memory_Bytes(MB)',
        'Node_Disk_Reads_Completed_Total(K)', 'Node_Disk_Reads_Merged_Total(K)', 'Node_Disk_Write_Time_Seconds_Total(s)',
        'Node_Disk_Writes_Merged_Total(K)',

    ]

    time_related_metrics = [
        'Process_Cpu_Seconds_Total(s)', 'Node_Cpu_Seconds_Total_Mode_System(s)', 'Container_Fs_Read_Seconds_Total(s)', 'Node_Cpu_Seconds_Total_Mode_Idel(s)',
        'Container_Fs_Io_Time_Seconds_Total(s)', 'Container_Fs_Io_Time_Weighted_Seconds_Total(s)', 'Container_Fs_Write_Seconds_Total(s)', 'Node_Cpu_Seconds_Total_Mode_User(s)',
        'Node_Pressure_Io_Stalled_Seconds_Total(s)', 'Node_Pressure_Io_Waiting_Seconds_Total(s)',
        'Node_Disk_Io_Time_Seconds_Total(s)','Node_Disk_Io_Time_Weighted_Seconds_Total(s)', 'Node_Disk_Read_Time_Seconds_Total(s)', 
        'Container_Fs_Io_Time_Seconds_Total(s)', 'Container_Fs_Io_Time_Weighted_Seconds_Total(s)', 'Container_Fs_Write_Seconds_Total(s)',  'Container_Fs_Read_Seconds_Total(s)',
        'Node_Cpu_Seconds_Total_Mode_Idel(s)', 'Node_Cpu_Seconds_Total_Mode_Iowait(s)', 'Node_Cpu_Seconds_Total_Mode_Softirq(s)', 'Node_Cpu_Seconds_Total_Mode_System(s)',
        'Node_Cpu_Seconds_Total_Mode_User(s)', 'Node_Schedstat_Running_Seconds_Total(s)', 'Process_Cpu_Seconds_Total(s)', 'Container_Cpu_System_Seconds_Total(s)', 
        'Container_Cpu_User_Seconds_Total(s)', 'Node_Pressure_Cpu_Waiting_Seconds_Total(s)', 'Node_Pressure_Memory_Stalled_Seconds_Total(s)', 'Node_Cpu_Seconds_Total_Mode_Nice(s)'
    ] 

    drop_total_metrics_list = [
        'Process_Cpu_Seconds_Total(s)', 'Node_Cpu_Seconds_Total_Mode_System(s)', 'Container_Fs_Read_Seconds_Total(s)', 'Node_Cpu_Seconds_Total_Mode_Idel(s)',
        'Container_Fs_Io_Time_Seconds_Total(s)', 'Container_Fs_Io_Time_Weighted_Seconds_Total(s)', 'Container_Fs_Write_Seconds_Total(s)', 'Node_Cpu_Seconds_Total_Mode_User(s)',
        'Container_Fs_Writes_Merged_Total', 'Process_Virtual_Memory_Bytes(GB)', 'Node_Intr_Total(K)', 'Container_Memory_Failures_Total',
    ]

    drop_list = drop_list + extra_info_list + time_related_metrics  + node_drop_list + useless_info_list #+ drop_total_metrics_list
    # drop_list

    dataset_X = dataset.drop(drop_list, axis=1).copy()
    # dataset_X

    dataset_Y = dataset['Active_Power(W)'].copy()
    # dataset_Y

    return TEST_SET_RATIO,dataset_X,dataset_Y


def main():
    test_set_ratio, df_X, df_Y = dataset_preprocess()
    parameter = {
        # 参数名字       #最小值  #最大值  # 类型
        'n_estimators': (200, 1000), #int
        'min_child_weight': (1, 13), #int
        'eta': (0.01, 0.20), # float
        'colsample_bytree': (0.01, 1.0), # float
        'max_depth': (1, 9), #int
        'subsample': (0.1, 1.0), # float
        'lambda': (0.10, 0.50), # float
        'learning_rate': (0.05, 0.15), # float
        'early_stopping_rounds': (25, 100), # int
        'n_important_features': (5, 36), # int
    }
    keys= list(parameter.keys())  #in python 3, you'll need `list(i.keys())`
    values= list(parameter.values())
    feature_names = list(df_X.columns)
    bounds = []
    for v in values:
        bounds.append(tuple(v))
    #print(bounds)
    #bounds = [(0.01,1.0), (1.0, 17.0), (2.0, 50.0), (2.0, 20.0), (1.0, 20.0), (50.0, 150.0)]
    result = differential_evolution(parameter_tuning_objective_func, maxiter=1000, popsize=1000, bounds=bounds, args=(test_set_ratio, keys, feature_names, df_X, df_Y))
    #result = differential_evolution(parameter_tuning_objective_func, bounds, args=(test_set_ratio, keys, values, df_X, df_Y))
    # result = differential_evolution(score_solver, bounds, args=(datasets,))
    #print(result.x, result.fun)
    cur_dir = pathlib.Path().resolve()
    result_file = cur_dir + '/result-' + str((datetime.now()).strftime("%d-%m-%Y-%H-%M-%S"))
    with open(result_file, 'w') as f:
        f.write(str(result.x))
        f.write(str(result.fun))

# l
# nohup /media/linux/anaconda3/envs/data_process/bin/python DE_xgboost_io_power_prediction.py > XGBoost-output-io-power-$(date "+%Y-%m-%d-%H:%M:%S")
if __name__ == '__main__':
    main()
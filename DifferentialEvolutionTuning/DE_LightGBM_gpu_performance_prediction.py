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
import lightgbm as lgb


from typing import Tuple
from sklearn.metrics import r2_score


def r2score(predt: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[str, float]:
    y = dtrain.get_label()
    #print(y)
    return 'R2', float(r2_score(y, predt))

def parameter_tuning_objective_func(X, *args):

    (test_set_ratio, keys, feature_names, df_X, df_Y) = args

    num_boost_round = int(X[keys.index('num_boost_round')])
    bagging_freq = int(X[keys.index('bagging_freq')])
    feature_fraction = np.clip(X[keys.index('feature_fraction')], 0.0, 1.0)
    bagging_fraction = np.clip(X[keys.index('bagging_fraction')], 0.0, 1.0)
    max_depth = int(X[keys.index('max_depth')])
    num_leaves = int(X[keys.index('num_leaves')])
    max_bin = int(X[keys.index('max_bin')])
    learning_rate = X[keys.index('learning_rate')]
    early_stopping_rounds = int(X[keys.index('early_stopping_rounds')])
    n_important_features = int(X[keys.index('n_important_features')])
    min_data_in_leaf = int(X[keys.index('min_data_in_leaf')])
    min_split_gain = np.clip(X[keys.index('min_split_gain')], 0.0, 1.0)
    lambda_l1 = np.clip(X[keys.index('lambda_l1')], 0.0, 1.0)
    lambda_l2 = np.clip(X[keys.index('lambda_l2')], 0.0, 1.0)

    args_list = [num_boost_round, bagging_freq, max_depth, feature_fraction, bagging_fraction, learning_rate, early_stopping_rounds, 
            n_important_features, num_leaves, max_bin, min_data_in_leaf, min_split_gain, lambda_l1, lambda_l2]

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
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': learning_rate,
        'feature_fraction': feature_fraction,
        'bagging_fraction': bagging_fraction,
        'bagging_freq': bagging_freq,
        'verbosity': -1,
        "max_depth": max_depth,
        "num_leaves": num_leaves,
        "max_bin": max_bin,
        'min_split_gain': min_split_gain,
        'lambda_l1': lambda_l1,
        'lambda_l2': lambda_l2,
    }

    dtrain = lgb.Dataset(data=X_train,label=y_train,free_raw_data=False)
    dtest = lgb.Dataset(data=X_test,label=y_test,free_raw_data=False)

    evals_result = {}
    bst = lgb.train(params = params,
        train_set = dtrain,
        num_boost_round=num_boost_round,
        valid_sets  = [dtest, dtrain],
        valid_names = ['eval', 'train'],
        callbacks = [
            lgb.record_evaluation(evals_result),
            lgb.log_evaluation(0),
            lgb.early_stopping(early_stopping_rounds,verbose=False)
        ]
        )


    feature_importances = np.array(list(bst.feature_importance()))
    sorted_idx = np.argsort(feature_importances)[::-1][:len(feature_importances)]
    feature_names = np.array(list(X.columns))[sorted_idx][:n_important_features]

    

    df_X = df_X.loc[:, feature_names]
    X = df_X.copy()
    y = df_Y.copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_set_ratio
    )
    

    sc = StandardScaler()

    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    data_dmatrix = lgb.Dataset(X_train,label=y_train,free_raw_data=False)
    lgb_cv_result = lgb.cv(params = params, train_set = data_dmatrix, num_boost_round  = num_boost_round, nfold = 20, stratified=False, eval_train_metric = False,)

    args_str = dict(zip(keys, args_list))
    
    #score = np.sqrt(mean_squared_error(dtest.get_label(), bst.predict(dtest)))
    score = lgb_cv_result['rmse-mean'][-1]

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
    filename = 'DataSet_GPU_Intensive'
    #filename = 'Power_Prediction_DataSet_GPU_Intensive_hashcat'
    #filename = 'Power_Prediction_DataSet_GPU_Intensive_END_BACKUP'
    dataset = pd.read_excel(dir + filename + '.xlsx')
    size1 = dataset.shape[0]
    #dataset = dataset[(dataset['Used_Time(s)'] <= 5.0) & (dataset['Used_Time(s)'] >= 1.0)]
    #dataset = dataset[(dataset['Used_Time(s)'] >= 1.0)]
    #dataset = dataset[~(dataset['Used_Time(s)'] >= 200.0)]
    #dataset = dataset[(dataset['Container_GPU_Utilization(%)'] > 0) & (dataset['Container_GPU_Utilization(%)'] < 50)]
    dataset = dataset[(dataset['Container_GPU_Utilization(%)'] > 0)]
    size2 = dataset.shape[0]

    print(size2)
    print('remove n samples: ', size1 - size2)

    drop_list = ['task_name', 'Activte_Energy(J)', 'Task_Energy_Concumption_Total(J)', 'Used_Time(s)', 'Net_Active_Power(W)', 'Frequency(Hz)',
        'Net_Active_Energy(J)', 'ABS_Net_Active_Power(W)', 'ABS_Net_Active_Energy(J)', 'TimeStamps', 'Container_GPU_Power_Usage(%)', 'Voltage(V)',
        'Container_GPU_Total_Energy_Consumption(J)', 'Container_GPU_Power_Total', 'Reactive_Energy(J)', 'Annual_Electricity_Consumption(KW/h)',
        'Current(A)', 'Power_Factor(PF)', 'Container_CPU_Request(%)', 'Container_Memory_Request(MB)', 'Container_GPU_Request', 'Container_IOPS_Request', #'program_name'
        'Node_Load15', 'Node_Load1', 'Container_Memory_Usage_Bytes(MB)', 'Container_Fs_Writes_Merged_Total',
        'lines_empty_num(source_code)','total_lines_num(source_code)','source_file_space_usage(K)',
        'lines_empty_num(kernel_code)','total_lines_num(kernel_code)','kernel_file_space_usage(K)',  
        #'lines_not_empty_num(kernel_code)','lines_not_empty_num(source_code)',
        #'program_name','lines_empty_num(assembly code)','total_lines_num(assembly code)',
        # 'Active_Power(W)', 'lines_not_empty_num(assembly code)', 'file_space_usage(K)',
        # 'Container_Memory_Working_Set_Bytes(MB)', 'Container_Memory_Max_Usage_Bytes(MB)', 'Container_Memory_Usage_Bytes(MB)',
        'p_name1', 'p_name2', 'Active_Power(W)',
        # 'Container_Memory_Mapped_File(MB)', 'Container_Memory_Working_Set_Bytes(MB)',
        # 'Node_Procs_Blocked', 
                'Container_Memory_Working_Set_Bytes(MB)', 'Container_Memory_Cached(MB)', 'Node_Load5', 'Node_Procs_Blocked',
        'Container_Memory_Mapped_File(MB)', 'Node_Intr_Total(K)',
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
        'L1-dcache-load-misses-rate(%)', 'dTLB-load-misses-rate(%)',
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
        'Node_Disk_Writes_Merged_Total(K)',  'Container_Memory_Max_Usage_Bytes(MB)', 
        'Container_Memory_Failures_Total',  'Container_Fs_Writes_Bytes_Total(MB)',
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

    drop_list = drop_list + time_related_metrics  + node_drop_list
    # drop_list

    dataset_X = dataset.drop(drop_list, axis=1).copy()
    # dataset_X

    dataset_Y = dataset['Used_Time(s)'].copy()
    # dataset_Y

    return TEST_SET_RATIO,dataset_X,dataset_Y


def main():
    test_set_ratio, df_X, df_Y = dataset_preprocess()
    parameter = {
        # 参数名字       #最小值  #最大值  # 类型
        'num_boost_round': (200, 1000), #int
        'bagging_freq': (0, 100), #int
        'max_depth': (1, 9), #int
        'feature_fraction': (0.05, 1.05), # float
        'bagging_fraction': (0.05, 1.05), # float
        'learning_rate': (0.05, 0.15), # float
        'early_stopping_rounds': (25, 100), # int
        'n_important_features': (5, 36), # int
        'num_leaves': (4, 32), # int
        'max_bin': (16, 256), # int
        'min_data_in_leaf': (1, 120), # int
        'min_split_gain': (0, 1.1), # float
        'lambda_l1': (0, 1.1), # float
        'lambda_l2': (0, 1.1), # float
    }
    keys= list(parameter.keys())  #in python 3, you'll need `list(i.keys())`
    values= list(parameter.values())
    feature_names = list(df_X.columns)
    bounds = []
    for v in values:
        bounds.append(tuple(v))
    #print(bounds)
    #bounds = [(0.01,1.0), (1.0, 17.0), (2.0, 50.0), (2.0, 20.0), (1.0, 20.0), (50.0, 150.0)]
    result = differential_evolution(parameter_tuning_objective_func, maxiter=500, bounds=bounds, args=(test_set_ratio, keys, feature_names, df_X, df_Y))
    #result = differential_evolution(parameter_tuning_objective_func, bounds, args=(test_set_ratio, keys, values, df_X, df_Y))
    # result = differential_evolution(score_solver, bounds, args=(datasets,))
    #print(result.x, result.fun)
    cur_dir = pathlib.Path().resolve()
    result_file = cur_dir + '/result-' + str((datetime.now()).strftime("%d-%m-%Y-%H-%M-%S"))
    with open(result_file, 'w') as f:
        f.write(str(result.x))
        f.write(str(result.fun))

# nohup /media/linux/anaconda3/envs/data_process/bin/python DE_LightGBM_gpu_performance_prediction.py > LightGBM-gpu_performance-output-$(date "+%Y-%m-%d-%H-%M-%S")
if __name__ == '__main__':
    main()
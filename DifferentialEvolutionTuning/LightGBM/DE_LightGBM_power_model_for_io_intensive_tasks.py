from matplotlib import axis
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, ensemble
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_log_error,
    median_absolute_error,
    mean_poisson_deviance,
    mean_gamma_deviance,
    mean_tweedie_deviance,
)
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
import argparse
from datetime import datetime, date, timedelta
from scipy.optimize import differential_evolution
import pathlib
import lightgbm as lgb
from typing import Tuple
from sklearn.metrics import r2_score


def r2score(predt: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[str, float]:
    y = dtrain.get_label()

    return "R2", float(r2_score(y, predt))


def parameter_tuning_objective_func(X, *args):
    (test_set_ratio, keys, feature_names, df_X, df_Y) = args

    """
    During the iterations of differential evolution, preprocess the generated parameters
    to ensure that each parameter falls within a legal and valid range.
    """
    num_boost_round = int(X[keys.index("num_boost_round")])
    bagging_freq = int(X[keys.index("bagging_freq")])
    feature_fraction = np.clip(X[keys.index("feature_fraction")], 0.0, 1.0)
    bagging_fraction = np.clip(X[keys.index("bagging_fraction")], 0.0, 1.0)
    max_depth = int(X[keys.index("max_depth")])
    num_leaves = int(X[keys.index("num_leaves")])
    max_bin = int(X[keys.index("max_bin")])
    learning_rate = X[keys.index("learning_rate")]
    early_stopping_rounds = int(X[keys.index("early_stopping_rounds")])
    n_important_features = int(X[keys.index("n_important_features")])
    min_data_in_leaf = int(X[keys.index("min_data_in_leaf")])
    min_split_gain = np.clip(X[keys.index("min_split_gain")], 0.0, 1.0)
    lambda_l1 = np.clip(X[keys.index("lambda_l1")], 0.0, 1.0)
    lambda_l2 = np.clip(X[keys.index("lambda_l2")], 0.0, 1.0)

    # Aggregate each parameter requiring adjustment
    args_list = [
        num_boost_round,
        bagging_freq,
        max_depth,
        feature_fraction,
        bagging_fraction,
        learning_rate,
        early_stopping_rounds,
        n_important_features,
        num_leaves,
        max_bin,
        min_data_in_leaf,
        min_split_gain,
        lambda_l1,
        lambda_l2,
    ]

    X = df_X.copy()
    y = df_Y.copy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_set_ratio)
    
    # performing preprocessing 
    sc = StandardScaler()

    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    """
    Specify and set model learning parameters, like learning rate and iteration count. 
    Fine-tune these for optimal model performance and improved learning efficiency.
    """
    params = {
        "task": "train",
        "boosting_type": "gbdt",
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": learning_rate,
        "feature_fraction": feature_fraction,
        "bagging_fraction": bagging_fraction,
        "bagging_freq": bagging_freq,
        "verbosity": -1,
        "max_depth": max_depth,
        "num_leaves": num_leaves,
        "max_bin": max_bin,
        "min_split_gain": min_split_gain,
        "lambda_l1": lambda_l1,
        "lambda_l2": lambda_l2,
    }

    dtrain = lgb.Dataset(data=X_train, label=y_train, free_raw_data=False)
    dtest = lgb.Dataset(data=X_test, label=y_test, free_raw_data=False)

    evals_result = {}

    """
    In the first iteration, the objective is to acquire a set of n features that exert the most significant influence.
    """
    bst = lgb.train(
        params=params,
        train_set=dtrain,
        num_boost_round=num_boost_round,
        valid_sets=[dtest, dtrain],
        valid_names=["eval", "train"],
        callbacks=[
            lgb.record_evaluation(evals_result),
            lgb.log_evaluation(0),
            lgb.early_stopping(early_stopping_rounds, verbose=False),
        ],
    )

    """
    Selection of the top n features exhibiting the most significant impact 
    on power consumption or performance of tasks.
    """
    feature_importances = np.array(list(bst.feature_importance()))
    sorted_idx = np.argsort(feature_importances)[::-1][: len(feature_importances)]
    feature_names = np.array(list(X.columns))[sorted_idx][:n_important_features]

    df_X = df_X.loc[:, feature_names]
    X = df_X.copy()
    y = df_Y.copy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_set_ratio)

    # performing preprocessing
    sc = StandardScaler()

    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    data_dmatrix = lgb.Dataset(X_train, label=y_train, free_raw_data=False)

    """
    In the second iteration, the model is trained utilizing a set of n features identified as exerting the most significant impact,
    and the evaluation metrics resulting from the training process are stored in a variable named "Score."
    """
    lgb_cv_result = lgb.cv(
        params=params,
        train_set=data_dmatrix,
        num_boost_round=num_boost_round,
        nfold=20,
        stratified=False,
        eval_train_metric=False,
    )

    args_str = dict(zip(keys, args_list))

    score = lgb_cv_result["rmse-mean"][-1]

    # Print intermediate result
    print(args_str, score)
    # print("The root mean squared error (RMSE) on test set: {:.4f}".format(score))
    # print("The mean squared error (MSE) on test set: {:.4f}".format(mean_squared_error(dtest.get_label(), bst.predict(dtest))))
    # print("The mean absolute error (MAE) on test set: {:.4f}".format(mean_absolute_error(dtest.get_label(), bst.predict(dtest))))
    # print("The mean absolute percentage error (MAPE) on test set: {:.4f}".format(mean_absolute_percentage_error(dtest.get_label(), bst.predict(dtest))))
    # print("The R square (R2) on test set: {:.4f}".format(np.sqrt(r2_score(dtest.get_label(), bst.predict(dtest)))))
    # print("The median_absolute_error (MEAE) on test set: {:.4f}".format(median_absolute_error(dtest.get_label(), bst.predict(dtest))))

    """
    'Score' here represents the quality of candidate solutions, 
    and the differential evolution algorithm optimizes this 'Score' to search for solutions to the problem. 
    In each generation, the algorithm generates new candidate solutions and decides whether to accept 
    or further improve them based on their scores. Through iterative refinement, 
    the algorithm aims to find the optimal solution to the problem by minimizing the 'Score'.
    """
    return score


def dataset_preprocess():
    # Determine the partition ratio for the training dataset and test dataset
    TEST_SET_RATIO = 0.5

    # Prepare the dataset
    file = "IO_ZONE_IO_Intensive.xlsx"
    dataset = pd.read_excel(file)

    """
    Prior to model training, we seek to eliminate extraneous features such as Power Factor (PF), Current (A),
    Working Hours Per Day, etc., and retain only approximately 60 parameters specified in the parameter list
    outlined in our research paper.
    """
    drop_list = ["Power Factor (PF)", "Current (A)", "Working Hours Per Day", "..."]

    # Determine the feature matrix
    dataset_X = dataset.drop(drop_list, axis=1).copy()

    # Determine the target variable
    dataset_Y = dataset["Active_Power(W)"].copy()

    return TEST_SET_RATIO, dataset_X, dataset_Y


def main():
    test_set_ratio, df_X, df_Y = dataset_preprocess()

    """
    Specify the range for each parameter to be adjusted during the process 
    of differential evolution hyperparameter tuning.
    """
    parameter = {
        # Parameter Name          # Min Value   # Max Value    # Type
        'num_boost_round':         (200,         1000),        # int
        'bagging_freq':            (0,           100),         # int
        'max_depth':               (1,           9),           # int
        'feature_fraction':        (0.05,        1.05),        # float
        'bagging_fraction':        (0.05,        1.05),        # float
        'learning_rate':           (0.01,        0.10),        # float
        'early_stopping_rounds':   (25,          100),         # int
        'n_important_features':    (5,           36),          # int
        'num_leaves':              (4,           156),         # int
        'max_bin':                 (16,          500),         # int
        'min_data_in_leaf':        (1,           120),         # int
        'min_split_gain':          (0,           1.1),         # float
        'lambda_l1':               (0,           1.1),         # float
        'lambda_l2':               (0,           1.1),         # float
    }

    keys = list(parameter.keys())  
    values = list(parameter.values())
    feature_names = list(df_X.columns)

    """
    Define the bounds for hyperparameter tuning. 
    The bounds take the form of a list of tuples, where each tuple represents 
    the range for a specific parameter. For example:
    bounds = [(0.01, 1.0), (1.0, 17.0), (2.0, 50.0), (2.0, 20.0), (1.0, 20.0), (50.0, 150.0)]
    """
    bounds = []
    for v in values:
        bounds.append(tuple(v))

    # Execute the differential evolution algorithm
    result = differential_evolution(
        parameter_tuning_objective_func,
        maxiter=1000,
        bounds=bounds,
        args=(test_set_ratio, keys, feature_names, df_X, df_Y),
    )

    # Save the final result
    cur_dir = pathlib.Path().resolve()
    result_file = (
        cur_dir + "/result-" + str((datetime.now()).strftime("%d-%m-%Y-%H-%M-%S"))
    )
    with open(result_file, "w") as f:
        f.write(str(result.x))
        f.write(str(result.fun))

if __name__ == "__main__":
    main()

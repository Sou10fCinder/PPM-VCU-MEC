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
    n_estimators = int(X[keys.index("n_estimators")])
    min_child_weight = int(X[keys.index("min_child_weight")])
    eta = X[keys.index("eta")]
    colsample_bytree = X[keys.index("colsample_bytree")]
    max_depth = int(X[keys.index("max_depth")])
    subsample = X[keys.index("subsample")]
    lambda_ = X[keys.index("lambda")]
    learning_rate = X[keys.index("learning_rate")]
    early_stopping_rounds = int(X[keys.index("early_stopping_rounds")])
    n_important_features = int(X[keys.index("n_important_features")])

    # Aggregate each parameter requiring adjustment
    args_list = [
        n_estimators,
        min_child_weight,
        eta,
        colsample_bytree,
        max_depth,
        subsample,
        lambda_,
        learning_rate,
        early_stopping_rounds,
        n_important_features,
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
        "n_estimators": n_estimators,  
        "min_child_weight": min_child_weight,  
        "eta": eta, 
        "colsample_bytree": colsample_bytree, 
        "max_depth": max_depth,  
        "subsample": subsample,  
        "lambda": lambda_,  
        "objective": "reg:squarederror",
        "learning_rate": learning_rate,  
        "predictor": "cpu_predictor",
        "verbosity": 0,
    }

    dtrain = xgb.DMatrix(data=X_train, label=y_train, feature_names=feature_names)
    dtest = xgb.DMatrix(data=X_test, label=y_test, feature_names=feature_names)

    evals_result = {}

    """
    In the first iteration, the objective is to acquire a set of n features that exert the most significant influence.
    """
    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=n_estimators,
        evals=[(dtest, "eval"), (dtrain, "train")],
        early_stopping_rounds=early_stopping_rounds,
        evals_result=evals_result,
        verbose_eval=False,
    )

    """
    Selection of the top n features exhibiting the most significant impact 
    on power consumption or performance of tasks.
    """
    feature_importances = np.array(
        list(bst.get_score(importance_type="weight").values())
    )
    sorted_idx = np.argsort(feature_importances)[::-1][: len(feature_importances)]
    feature_names = np.array(list(bst.get_score(importance_type="weight").keys()))[
        sorted_idx
    ][:n_important_features]

    df_X = df_X.loc[:, feature_names]
    X = df_X.copy()
    y = df_Y.copy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_set_ratio)

    # performing preprocessing
    sc = StandardScaler()

    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    data_dmatrix = xgb.DMatrix(data=X_train, label=y_train, feature_names=feature_names)

    """
    In the second iteration, the model is trained utilizing a set of n features identified as exerting the most significant impact,
    and the evaluation metrics resulting from the training process are stored in a variable named "Score."
    """
    xgb_cv = xgb.cv(
        dtrain=data_dmatrix,
        params=params,
        num_boost_round=n_estimators,
        verbose_eval=False,
        early_stopping_rounds=early_stopping_rounds,
        nfold=20,
        metrics="rmse",
    )  

    args_str = dict(zip(keys, args_list))

    score = xgb_cv[-1:].values[0][2]

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
    dataset_Y = dataset["Used_Time(s)"].copy()

    return TEST_SET_RATIO, dataset_X, dataset_Y


def main():
    test_set_ratio, df_X, df_Y = dataset_preprocess()

    """
    Specify the range for each parameter to be adjusted during the process 
    of differential evolution hyperparameter tuning.
    """
    parameter = {
        # Parameter Name          # Min Value  # Max Value   # Type
        'n_estimators':            (200,       1000),        # int
        'min_child_weight':        (1,         13),          # int
        'eta':                     (0.01,      0.20),        # float
        'colsample_bytree':        (0.01,      1.0),         # float
        'max_depth':               (1,         9),           # int
        'subsample':               (0.1,       1.0),         # float
        'lambda':                  (0.10,      0.50),        # float
        'learning_rate':           (0.01,      0.10),        # float
        'early_stopping_rounds':   (25,        100),         # int
        'n_important_features':    (5,         36),          # int
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
        popsize=1000,
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

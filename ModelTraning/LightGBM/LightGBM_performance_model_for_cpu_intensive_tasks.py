import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    median_absolute_error,
)
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import xgboost as xgb
import shap
import lightgbm as lgb
from typing import Tuple
from sklearn.metrics import r2_score


def main():
    # Determine the partition ratio for the training dataset and test dataset
    TEST_SET_RATIO = 0.5

    # Prepare the dataset
    file = "DataSet_CPU_Intensive.xlsx"
    dataset = pd.read_excel(file)

    """
    Prior to model training, we seek to eliminate extraneous features such as Power Factor (PF), Current (A),
    Working Hours Per Day, etc., and retain only approximately 60 parameters specified in the parameter list
    outlined in our research paper.
    """
    drop_list = ["Used_Time(s)", "Power Factor (PF)", "Current (A)", "Working Hours Per Day", "..."]

    # Determine the feature matrix
    dataset_X = dataset.drop(drop_list, axis=1).copy()

    # Determine the target variable
    dataset_Y = dataset["Used_Time(s)"].copy()

    X = dataset_X.copy()
    y = dataset_Y.copy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SET_RATIO)

    # performing preprocessing
    sc = StandardScaler()

    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    """
    Specify and set model parameters, including learning rate and iteration count, among others.
    All these parameters have been meticulously tuned through the application of a previously employed 
    differential evolution algorithm for hyperparameter adjustment.
    """
    params = {
        "task": "train",
        "boosting_type": "gbdt",
        "objective": "regression",
        "metric": "rmse",
        "num_boost_round": 782,
        "bagging_freq": 91,
        "max_depth": 7,
        "feature_fraction": 0.9111436990407021,
        "bagging_fraction": 0.9634521559193232,
        "learning_rate": 0.09321833733203178,
        "early_stopping_rounds": 83,
        "n_important_features": 20,
        "num_leaves": 25,
        "max_bin": 125,
        "min_data_in_leaf": 17,
        "min_split_gain": 0.7701342905252989,
        "lambda_l1": 0.4564399584331826,
        "lambda_l2": 0.09638986061422983,
    }

    n_estimators = params["num_boost_round"]
    early_stopping_rounds = params["early_stopping_rounds"]
    n_important_features = params["n_important_features"]
    params.pop("n_important_features")

    dtrain = lgb.Dataset(data=X_train, label=y_train)
    dtest = lgb.Dataset(data=X_test, label=y_test)

    evals_result = {}

    """
    In the first iteration, the objective is to acquire a set of n features that exert the most significant influence.
    """
    bst = lgb.train(
        params=params,
        train_set=dtrain,
        num_boost_round=n_estimators,
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

    dataset_X = dataset_X.loc[:, feature_names]
    X = dataset_X.copy()
    y = dataset_Y.copy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SET_RATIO)

    # performing preprocessing
    sc = StandardScaler()

    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    dtrain = lgb.Dataset(data=X_train, label=y_train)
    dtest = lgb.Dataset(data=X_test, label=y_test)

    evals_result = {}

    """
    In the second iteration, the model is trained utilizing a set of n features identified as exerting the most significant impact,
    and the evaluation metrics resulting from the training process are stored in a variable named "Score."
    """
    bst = lgb.train(
        params=params,
        train_set=dtrain,
        num_boost_round=n_estimators,
        valid_sets=[dtest, dtrain],
        valid_names=["eval", "train"],
        callbacks=[
            lgb.record_evaluation(evals_result),
            lgb.log_evaluation(0),
            lgb.early_stopping(early_stopping_rounds, verbose=False),
        ],
    )

    """
    Presenting the post-training performance of a machine learning model using multiple metric indicators.
    """
    rmse = np.sqrt(mean_squared_error(y_test, bst.predict(X_test)))
    print("The root mean squared error (RMSE) on test set: {:.4f}".format(rmse))
    print(
        "The mean squared error (MSE) on test set: {:.4f}".format(
            mean_squared_error(y_test, bst.predict(X_test))
        )
    )
    print(
        "The mean absolute error (MAE) on test set: {:.4f}".format(
            mean_absolute_error(y_test, bst.predict(X_test))
        )
    )
    print(
        "The mean absolute percentage error (MAPE) on test set: {:.4f}".format(
            mean_absolute_percentage_error(y_test, bst.predict(X_test))
        )
    )
    print(
        "The R square (R2) on test set: {:.4f}".format(
            r2_score(y_test, bst.predict(X_test))
        )
    )
    print(
        "The median_absolute_error (MEAE) on test set: {:.4f}".format(
            median_absolute_error(y_test, bst.predict(X_test))
        )
    )

    model_name = "model_name"
    bst.save_model(model_name + ".json")

    # Perform cross-validation and output the results
    data_dmatrix = lgb.Dataset(X_train, label=y_train)
    lgb_cv_result = lgb.cv(
        params=params,
        train_set=data_dmatrix,
        num_boost_round=n_estimators,
        nfold=10,
        stratified=False,
        eval_train_metric=False,
        verbose_eval=False,
    )

    score = lgb_cv_result["rmse-mean"][-1]
    print("score", score)

    """
    SHAP (SHapley Additive exPlanations) determines feature importance by calculating Shapley values for each feature
    through permutations of all possible feature combinations. Specifically, for each sample, SHAP values account for
    the impact of different feature combinations on the model output. The relative contribution of each feature is
    established by taking a weighted average across these combinations.
    """
    explainer = shap.TreeExplainer(bst)
    shap_values = explainer(X)
    feature_names = shap_values.feature_names
    shap_df = pd.DataFrame(shap_values.values, columns=X.columns)
    vals = np.abs(shap_df.values).mean(0)
    shap_importance = pd.DataFrame(
        list(zip(feature_names, vals)), columns=["col_name", "feature_importance_vals"]
    )
    shap_importance.sort_values(
        by=["feature_importance_vals"], ascending=False, inplace=True
    )

    # Save the results of the feature importance array
    shap_importance.to_csv(model_name + ".csv")


if __name__ == "__main__":
    main()

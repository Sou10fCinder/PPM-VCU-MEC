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
    file = "DataSet_GPU_Intensive.xlsx"
    dataset = pd.read_excel(file)

    """
    Prior to model training, we seek to eliminate extraneous features such as Power Factor (PF), Current (A),
    Working Hours Per Day, etc., and retain only approximately 60 parameters specified in the parameter list
    outlined in our research paper.
    """
    drop_list = [
        "Used_Time(s)",
        "Power Factor (PF)",
        "Current (A)",
        "Working Hours Per Day",
        "...",
    ]

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
        "num_boost_round": 696,
        "bagging_freq": 74,
        "max_depth": 5,
        "feature_fraction": 0.8675456544023968,
        "bagging_fraction": 0.8784643153609198,
        "learning_rate": 0.09405696789233671,
        "early_stopping_rounds": 44,
        "n_important_features": 11,
        "num_leaves": 4,
        "max_bin": 84,
        "min_data_in_leaf": 7,
        "min_split_gain": 0.36781089191574573,
        "lambda_l1": 0.4084609015864684,
        "lambda_l2": 0.20658725050370058,
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
    In the second iteration, the model is trained utilizing a set of n features identified 
    as exerting the most significant impact
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
    print(
        "The root mean squared error (RMSE) on test set: {:.4f}".format(
            np.sqrt(mean_squared_error(y_test, bst.predict(X_test)))
        )
    )
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

    # Perform cross-validation and output the results
    data_dmatrix = lgb.Dataset(X_train, label=y_train)
    lgb_cv_result = lgb.cv(
        params=params,
        train_set=data_dmatrix,
        num_boost_round=n_estimators,
        nfold=20,
        stratified=False,
        eval_train_metric=False,
    )

    score = lgb_cv_result["rmse-mean"][-1]
    print("score", score)

    # plot the learning curve
    fig = plt.figure(figsize=(6, 6))
    plt.subplot(1, 1, 1)
    plt.title("RMSE")
    plt.plot(
        np.arange(bst.best_iteration) + 1,
        evals_result["train"]["rmse"][: bst.best_iteration],
        "b-",
        label="Training Set RMSE",
    )
    plt.plot(
        np.arange(bst.best_iteration) + 1,
        evals_result["eval"]["rmse"][: bst.best_iteration],
        "r-",
        label="Test Set RMSE",
    )
    plt.legend(loc="upper right")
    plt.xlabel("Boosting Iterations")
    plt.ylabel("RMSE")
    fig.tight_layout()

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

    print(shap_importance)

    # Plot a chart of the key feature array to illustrate the outcomes of its importance analysis.
    pos = np.arange(shap_importance.shape[0]) + 0.5
    fig = plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.barh(pos, shap_importance.loc[:, "feature_importance_vals"], align="center")
    plt.yticks(
        pos,
        np.array(shap_importance.loc[:, "col_name"]),
    )
    plt.title("Feature Importance")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

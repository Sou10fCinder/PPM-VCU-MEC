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
import seaborn as sns
import xgboost as xgb
import shap
from typing import Tuple
from sklearn.metrics import r2_score

def main():
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
    drop_list = ["Active_Power(W)", "Power Factor (PF)", "Current (A)", "Working Hours Per Day", "..."]

    # Determine the feature matrix
    dataset_X = dataset.drop(drop_list, axis=1).copy()

    # Determine the target variable
    dataset_Y = dataset["Active_Power(W)"].copy()


    X = dataset_X.copy()
    y = dataset_Y.copy()


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SET_RATIO)

    """
    Specify and set model parameters, including learning rate and iteration count, among others.
    All these parameters have been meticulously tuned through the application of a previously employed 
    differential evolution algorithm for hyperparameter adjustment.
    """
    params = {
        "n_estimators": 777,
        "min_child_weight": 4,
        "eta": 0.05496937604915489,
        "colsample_bytree": 0.7006833426785724,
        "max_depth": 4,
        "subsample": 0.547061410889487,
        "lambda": 0.46892295995198696,
        "learning_rate": 0.014623989130071173,
        "early_stopping_rounds": 86,
        "n_important_features": 9,
    } 

    n_estimators = params["n_estimators"]
    early_stopping_rounds = params["early_stopping_rounds"]
    n_important_features = params["n_important_features"]
    params.pop("n_estimators")
    params.pop("early_stopping_rounds")
    params.pop("n_important_features")


    dtrain = xgb.DMatrix(data=X_train, label=y_train, feature_names=dataset_X.columns)
    dtest = xgb.DMatrix(data=X_test, label=y_test, feature_names=dataset_X.columns)

    evals_result = {}

    """
    In the first iteration, the objective is to acquire a set of n features that exert the most significant influence.
    """
    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=n_estimators,
        evals=[(dtest, "eval"), (dtrain, "train")],
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


    dataset_X = dataset_X.loc[:, feature_names]
    X = dataset_X.copy()
    y = dataset_Y.copy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SET_RATIO)


    dtrain = xgb.DMatrix(data=X_train, label=y_train, feature_names=feature_names)
    dtest = xgb.DMatrix(data=X_test, label=y_test, feature_names=feature_names)

    evals_result = {}

    """
    In the second iteration, the model is trained utilizing a set of n features identified 
    as exerting the most significant impact
    """
    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=n_estimators,
        evals=[(dtest, "eval"), (dtrain, "train")],
        early_stopping_rounds=early_stopping_rounds,
        evals_result=evals_result,
    )

    """
    Presenting the post-training performance of a machine learning model using multiple metric indicators.
    """
    rmse = np.sqrt(mean_squared_error(dtest.get_label(), bst.predict(dtest)))
    print("The root mean squared error (RMSE) on test set: {:.4f}".format(rmse))
    print(
        "The mean squared error (MSE) on test set: {:.4f}".format(
            mean_squared_error(dtest.get_label(), bst.predict(dtest))
        )
    )
    print(
        "The mean absolute error (MAE) on test set: {:.4f}".format(
            mean_absolute_error(dtest.get_label(), bst.predict(dtest))
        )
    )
    print(
        "The mean absolute percentage error (MAPE) on test set: {:.4f}".format(
            mean_absolute_percentage_error(dtest.get_label(), bst.predict(dtest))
        )
    )
    R2 = r2_score(dtest.get_label(), bst.predict(dtest))
    print("The R square (R2) on test set: {:.4f}".format(R2))
    print(
        "The median_absolute_error (MEAE) on test set: {:.4f}".format(
            median_absolute_error(dtest.get_label(), bst.predict(dtest))
        )
    )

    # Perform cross-validation and output the results
    data_dmatrix = xgb.DMatrix(data=X_train, label=y_train, feature_names=feature_names)
    xgb_cv = xgb.cv(
        dtrain=data_dmatrix,
        params=params,
        num_boost_round=n_estimators,
        nfold=20,
        verbose_eval=True,
        early_stopping_rounds=early_stopping_rounds,
    )  


    score = xgb_cv[-1:].values[0][2]
    print("score", score)


    # plot the learning curve
    fig = plt.figure(figsize=(6, 6))
    plt.subplot(1, 1, 1)
    plt.title("RMSE")
    plt.plot(
        np.arange(n_estimators) + 1,
        evals_result["train"]["rmse"],
        "b-",
        label="Training Set RMSE",
    )
    plt.plot(
        np.arange(n_estimators) + 1,
        evals_result["eval"]["rmse"],
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


    # Plot a chart of the key feature array to illustrate the outcomes of its importance analysis
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

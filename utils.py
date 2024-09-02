import category_encoders as ce
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

### IMPUTATION ###
from hyperimpute.plugins.imputers import Imputers

# Generic time series imputation
import pypots
from pypots.optim import Adam
from pypots.imputation import CSDI
from pypots.utils.metrics import calc_mae, calc_rmse

### EVALUATION ###

# ML efficiency
from autogluon.tabular import TabularPredictor


def standardise(df, label, scaler=None):
    """
    Standardizes the feature columns in a DataFrame, excluding the label column.

    Parameters:
    df:      pd.DataFrame, the DataFrame containing features and the label column.
    label:   str, the name of the label column to exclude from standardization.
    scaler:  StandardScaler, optional, an instance of StandardScaler. If None, a new StandardScaler will be created.

    Returns:
    pd.DataFrame: A DataFrame with standardized feature columns and the original label column.
    """
    # Separate features and label
    features = df.drop(columns=[label])
    target = df[label]

    # Use provided scaler or create a new one
    if scaler is None:
        scaler = StandardScaler()

    # Apply standardisation to the features
    features_scaled = pd.DataFrame(
        scaler.fit_transform(features), columns=features.columns, index=features.index
    )

    # Combine the scaled features with the label column
    df_scaled = pd.concat([features_scaled, target], axis=1)

    return df_scaled, scaler


def cal_ml_efficiency(train, test, label, eval_metric="f1_macro"):
    """
    Evaluate machine learning efficiency using AutoGluon's TabularPredictor.

    TODO: Try a few other models and evaluation metrics.

    Parameters:
    - train:        pd.DataFrame, the training dataset.
    - test:         pd.DataFrame, the testing dataset.
    - label:        str, the label column.
    - eval_metric:  str, the evaluation metric to use.

    Returns:
    - dict, evaluation results.
    """
    predictor = TabularPredictor(label=label, eval_metric=eval_metric)

    # Define the hyperparameters to use only NN_Torch
    hyperparameters = {
        "NN_TORCH": {
            "num_layers": 3,  # Number of layers in the model
            "hidden_size": 128,  # Size of the hidden layers
        },
        # TODO: You can add other models or hyperparameters here, e.g.:
    }

    # Fit the model with only NN_Torch
    predictor.fit(train, hyperparameters=hyperparameters, presets="best_quality")

    return predictor.evaluate(test)


def cal_metrics(original, imputed, metric="all"):
    """
    Calculate imputation quality metrics for given original and imputed datasets.

    Parameters:
    - original: pd.DataFrame, the original dataset with missing values.
    - imputed:  pd.DataFrame, the imputed dataset.
    - metric:   str, the metric to calculate ('mae' or 'rmse').

    Returns:
    - float, the calculated metric value.
    """
    valid_rows = ~original.isnull().any(axis=1)
    ori_np = original[valid_rows].to_numpy()
    imp_np = imputed[valid_rows].to_numpy()

    metrics = {
        "mae": calc_mae,
        "rmse": calc_rmse,
    }

    if metric == "all":
        return {name: func(ori_np, imp_np) for name, func in metrics.items()}
    elif metric in metrics:
        return metrics[metric](ori_np, imp_np)
    else:
        raise ValueError(f"Invalid metric '{metric}'. Choose 'mae', 'rmse', or 'all'.")


def gene_missing_at_random(data, missing_rate, label_col, random_state=42):
    """
    Generate missing data at random in a DataFrame.

    Parameters:
    - data:         pd.DataFrame, the input DataFrame.
    - missing_rate: float, the proportion of data to be set as missing (0 <= missing_rate <= 1).
    - label_col:    str, the name of the label column, which should not be set as missing.
    - random_state: int, the random seed to use.

    Returns:
    - pd.DataFrame, the original DataFrame.
    - pd.DataFrame, with missing values introduced.
    """
    np.random.seed(random_state)

    # Separate the feature columns and the label column
    feature_columns = data.drop(columns=[label_col])
    label_column = data[label_col]

    # Masking the feature columns with missing values
    mask = np.random.rand(*feature_columns.shape) < missing_rate
    features_with_missing = feature_columns.mask(mask)

    return data, pd.concat([features_with_missing, label_column], axis=1)


def impute_missing_values(
    algo,
    df_missing,
    train_indices,
    test_indices,
    random_state=42,
    categorical_cols=None,
):
    """
    Impute missing values in a DataFrame using the specified algorithm.

    Parameters:
    - algo:             str, the imputation algorithm to use.
    - df_missing:       pd.DataFrame, the original DataFrame with missing values.
    - train_indices:    list, the indices of the training set.
    - test_indices:     list, the indices of the testing set.
    - random_state:     int, the random seed to use.
    - categorical_cols: list, the names of the categorical columns.


    Returns:
    - pd.DataFrame, the imputed train and test DataFrames.
    """
    print("Available imputing algorithms:\n" + "\n".join(Imputers().list()))

    imputer = Imputers().get(algo, random_state=random_state)
    df_imp = imputer.fit_transform(df_missing.copy())

    # assign the column names back
    df_imp.columns = df_missing.columns

    if categorical_cols:
        df_imp[categorical_cols] = df_imp[categorical_cols].round()

    # Split the data into training and testing sets
    train_imp = df_imp.loc[train_indices]
    test_imp = df_imp.loc[test_indices]

    return train_imp, test_imp

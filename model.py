"""
NumPy House Price Regression

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - impute_nan_with_mean
def impute_nan_with_mean(X):
    """Replace every NaN in X with that column's nan-aware mean (all-NaN cols -> 0).

    Args:
        X: (N, F) array-like of floats, may contain NaN.

    Returns:
        (N, F) float ndarray with no NaNs.
    """
    mean_v = np.nan_to_num(np.nanmean(X, axis=0), 0.0)
    return np.where(np.isnan(X), mean_v, X)

# Step 2 - compute_iqr_bounds
def compute_iqr_bounds(X, k=1.5):
    q1, q3 = np.quantile(X, [0.25, 0.75], axis=0)
    iqr = k * (q3 - q1)
    return (q1 - iqr, q3 + iqr)

# Step 3 - clip_columns
def clip_columns(X, lower, upper):
    return np.clip(X, lower, upper)

# Step 4 - make_ratio_feature
def make_ratio_feature(numerator, denominator, eps=1e-8):
    return numerator / (denominator + eps)

# Step 5 - append_column
def append_column(X, col):
    return np.hstack([X, col[:, None]])

# Step 6 - one_hot_encode
def one_hot_encode(labels):
    unique, idx = np.unique(labels, return_inverse=True)
    one_hot = np.eye(unique.shape[0])
    return one_hot[idx]

# Step 7 - fit_standardizer
def fit_standardizer(X):
    std_v = X.std(axis=0)
    return X.mean(axis=0), np.where(std_v==0, 1.0, std_v)

# Step 8 - apply_standardizer
def apply_standardizer(X, mean, std):
    return (X - mean) / std

# Step 9 - add_bias_column
def add_bias_column(X):
    return np.hstack([np.ones((X.shape[0], 1)), X])

# Step 10 - make_shuffled_indices
def make_shuffled_indices(n_samples, seed):
    return np.random.default_rng(seed).permutation(n_samples)

# Step 11 - partition_indices
def partition_indices(indices, train_ratio, val_ratio):
    n = indices.shape[0]
    train_end = int(n*train_ratio)
    val_end = int(n*val_ratio) + train_end

    return \
    (
        indices[:train_end],
        indices[train_end:val_end],
        indices[val_end:]
    )

# Step 12 - subset_xy
def subset_xy(X, y, indices):
    return X[indices], y[indices]

# Step 13 - ols_fit
def ols_fit(X, y):
    return np.linalg.inv(X.T @ X) @ X.T @ y

# Step 14 - ols_predict
def ols_predict(X, theta):
    return np.dot(X, theta)

# Step 15 - mean_absolute_error
def mean_absolute_error(y_true, y_pred):
    return np.abs(y_true - y_pred).mean()

# Step 16 - root_mean_squared_error
def root_mean_squared_error(y_true, y_pred):
    """Compute root mean squared error between targets and predictions.

    Args:
        y_true (np.ndarray): Ground-truth targets, shape (N,).
        y_pred (np.ndarray): Predicted targets, shape (N,).

    Returns:
        float: RMSE value.
    """
    return np.sqrt(((y_true - y_pred)**2).mean())

# Step 17 - r_squared
def r_squared(y_true, y_pred):
    ss_res = ((y_true - y_pred)**2).sum()
    ss_tot = ((y_true - y_true.mean())**2).sum()
    if (ss_tot == 0): return 0
    return 1 - (ss_res / ss_tot)

# Step 18 - residual_summary
def residual_summary(y_true, y_pred):
    r = y_true - y_pred
    return \
    {
        "mean" : r.mean(),
        "std" : r.std(),
        "median_abs" : np.median(np.abs(r))
    }

# Step 19 - prepare_cleaned_features (not yet solved)
# TODO: implement

# Step 20 - assemble_feature_matrix (not yet solved)
# TODO: implement

# Step 21 - make_train_val_test (not yet solved)
# TODO: implement

# Step 22 - standardize_and_add_bias (not yet solved)
# TODO: implement

# Step 23 - evaluate_predictions (not yet solved)
# TODO: implement

# Step 24 - house_price_pipeline (not yet solved)
# TODO: implement


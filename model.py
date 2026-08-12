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
    col = col.reshape(X.shape[0], -1)
    return np.hstack([X, col])

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
    np.random.seed(seed)
    return np.random.permutation(n_samples)

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
    w, _, _, _ = np.linalg.lstsq(X, y)
    return w

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

# Step 19 - prepare_cleaned_features
def prepare_cleaned_features(X, iqr_k=1.5):
    """Impute NaNs then IQR-clip columns to produce a clean numeric matrix.

    Args:
        X: (N, F) array-like of floats, may contain NaN.
        iqr_k: IQR multiplier passed to compute_iqr_bounds (default 1.5).

    Returns:
        (N, F) float ndarray with no NaNs, columns clipped to IQR bounds.
    """
    X = impute_nan_with_mean(X)
    lower_X, upper_X = compute_iqr_bounds(X, iqr_k)
    return clip_columns(X, lower_X, upper_X)

# Step 20 - assemble_feature_matrix
def assemble_feature_matrix(X_num, ratio_num_idx, ratio_den_idx, cat_labels=None):
    ratio_col = make_ratio_feature(X_num[:, ratio_num_idx], X_num[:, ratio_den_idx])
    ret = append_column(X_num, ratio_col)
    if (cat_labels is not None):
        one_hot = one_hot_encode(cat_labels)
        ret = append_column(ret, one_hot)

    return ret

# Step 21 - make_train_val_test
def make_train_val_test(X, y, train_ratio, val_ratio, seed):
    n_samples = X.shape[0]
    idx = make_shuffled_indices(n_samples, seed)
    train_idx, val_idx, test_idx = partition_indices(idx, train_ratio, val_ratio)

    x_train, y_train = subset_xy(X, y, train_idx)
    x_val, y_val = subset_xy(X, y, val_idx)
    x_test, y_test = subset_xy(X, y, test_idx)

    return \
    {
        "X_train" : x_train,
        "y_train" : y_train,
        "X_val" : x_val,
        "y_val" : y_val,
        "X_test" : x_test,
        "y_test" : y_test
    }

# Step 22 - standardize_and_add_bias
def standardize_and_add_bias(splits):
    train_x = splits["X_train"]
    val_x = splits["X_val"]
    test_x = splits["X_test"]

    mean_x, std_x = fit_standardizer(train_x)
    train_x = add_bias_column(apply_standardizer(train_x, mean_x, std_x))
    val_x = add_bias_column(apply_standardizer(val_x, mean_x, std_x))
    test_x = add_bias_column(apply_standardizer(test_x, mean_x, std_x))

    std_splits = splits.copy()

    std_splits["X_train"] = train_x
    std_splits["X_val"] = val_x
    std_splits["X_test"] = test_x
    
    return std_splits, mean_x, std_x

# Step 23 - evaluate_predictions
def evaluate_predictions(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    r2 = r_squared(y_true, y_pred)
    res_summary = residual_summary(y_true, y_pred)
    return \
    {
        "mae" : mae,
        "rmse" : rmse,
        "r2" : r2,
        "residual_summary" : res_summary
    }

# Step 24 - house_price_pipeline
def house_price_pipeline(X, y, ratio_num_idx, ratio_den_idx, cat_labels=None, train_ratio=0.7, val_ratio=0.15, seed=42, iqr_k=1.5):
    X = prepare_cleaned_features(X, iqr_k)
    X = assemble_feature_matrix(X, ratio_num_idx, ratio_den_idx, cat_labels)
    splits = make_train_val_test(X, y, train_ratio, val_ratio, seed)
    std_splits, _, _ = standardize_and_add_bias(splits)
    w = ols_fit(std_splits["X_train"], std_splits["y_train"])

    y_train_pred = ols_predict(std_splits["X_train"], w)
    y_val_pred = ols_predict(std_splits["X_val"], w)
    y_test_pred = ols_predict(std_splits["X_test"], w)

    train_metrics = evaluate_predictions(std_splits["y_train"], y_train_pred)
    val_metrics = evaluate_predictions(std_splits["y_val"], y_val_pred)
    test_metrics = evaluate_predictions(std_splits["y_test"], y_test_pred)

    return \
    {
        "theta" : w,
        "y_test" : std_splits["y_test"],
        "y_test_pred" : y_test_pred,
        "test_metrics" : test_metrics,
        "val_metrics" : val_metrics
    }


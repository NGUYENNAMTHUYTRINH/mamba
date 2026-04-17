import numpy as np, torch
from sklearn.preprocessing import StandardScaler
import joblib, os

def make_windows_for_station(features: np.ndarray, target: np.ndarray, lookback: int, horizon: int):
    """
    Produce sequence windows from a single continuous time-series (e.g., from one station).
    features shape: (seq_len, num_features)
    target shape: (seq_len,)
    Returns:
    X: (num_windows, lookback, num_features)
    y: (num_windows, horizon)
    """
    X, y = [], []
    # Ensure arrays are 2D/1D properly
    if len(features.shape) == 1:
        features = features.reshape(-1, 1)
    
    n_samples = len(features)
    for i in range(n_samples - lookback - horizon + 1):
        X.append(features[i : i + lookback, :])
        y.append(target[i + lookback : i + lookback + horizon])
    
    if len(X) == 0:
        return np.empty((0, lookback, features.shape[1])), np.empty((0, horizon))
    return np.array(X), np.array(y)

def make_windows_grouped(df, feature_cols, target_col, lookback, horizon, group_col="location_key"):
    """
    Group standard operations by location to avoid sequence overlap
    """
    X_all, y_all = [], []
    for location, group in df.groupby(group_col):
        # Sort by time just in case
        group = group.sort_values("ts_utc")
        if len(group) <= lookback + horizon: continue
            
        features = group[feature_cols].values
        target = group[target_col].values
        X, y = make_windows_for_station(features, target, lookback, horizon)
        
        if len(X) > 0:
            X_all.append(X)
            y_all.append(y)
            
    if not X_all:
        return np.array([]), np.array([])
    return np.concatenate(X_all, axis=0), np.concatenate(y_all, axis=0)

def make_windows_grouped_with_locids(df, feature_cols, target_col, lookback, horizon, loc2id, group_col="location_key"):
    """
    Like make_windows_grouped but also returns an integer location-ID array (one per window).
    loc2id: dict mapping location_key (str) -> int index
    Returns:
        X        : (N, lookback, num_features)
        y        : (N, horizon)
        loc_ids  : (N,)  int64 — index into the embedding table
    """
    X_all, y_all, loc_id_all = [], [], []
    for location, group in df.groupby(group_col):
        group = group.sort_values("ts_utc")
        if len(group) <= lookback + horizon:
            continue
        features = group[feature_cols].values
        target   = group[target_col].values
        X, y = make_windows_for_station(features, target, lookback, horizon)
        if len(X) > 0:
            X_all.append(X)
            y_all.append(y)
            loc_id_all.append(np.full(len(X), loc2id[location], dtype=np.int64))
    if not X_all:
        return np.array([]), np.array([]), np.array([])
    return (
        np.concatenate(X_all,      axis=0),
        np.concatenate(y_all,      axis=0),
        np.concatenate(loc_id_all, axis=0),
    )

def scale_features(df, feature_cols, target_col):
    """Scale features and target independently."""
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()
    
    # Fit y_scaler on original unscaled target data FIRST
    y_scaler.fit(df[[target_col]])
    
    # Scale all features (which includes target_col)
    df[feature_cols] = X_scaler.fit_transform(df[feature_cols])
    
    # df[target_col] is now scaled by X_scaler. We don't overwrite it. 
    # The neural network will train on this scaled target, and y_scaler 
    # will correctly inverse_transform it back to original ranges later.
    
    return df, X_scaler, y_scaler

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))
def mape(y_true, y_pred, eps=1e-8):
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)

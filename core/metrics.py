"""
core/metrics.py
---------------
Tính toán các chỉ số đánh giá model dùng chung cho Mamba, LSTM, TFT.

Hàm chính:
- compute_metrics(y_true, y_pred) → dict với mae, rmse, r2, mse
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Tính MAE, RMSE, R², MSE giữa giá trị thực và dự đoán.

    Parameters
    ----------
    y_true : array-like, shape (N,) hoặc (N, H)
        Giá trị thực tế (đã denormalize về scale gốc).
    y_pred : array-like, shape (N,) hoặc (N, H)
        Giá trị dự đoán (đã denormalize về scale gốc).

    Returns
    -------
    dict với các key: "mae", "rmse", "r2", "mse"
    """
    y_true = np.asarray(y_true, dtype=np.float32).flatten()
    y_pred = np.asarray(y_pred, dtype=np.float32).flatten()

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true và y_pred phải cùng số phần tử, "
            f"nhưng nhận được {len(y_true)} và {len(y_pred)}."
        )
    if len(y_true) == 0:
        raise ValueError("y_true và y_pred không được rỗng.")

    mse  = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))

    return {"mae": mae, "rmse": rmse, "r2": r2, "mse": mse}


def denormalize(arr: np.ndarray, mean: float, std: float) -> np.ndarray:
    """Chuyển array từ normalized space về scale gốc.

    Parameters
    ----------
    arr  : array đã được normalize bằng (x - mean) / std
    mean : giá trị mean dùng khi normalize
    std  : giá trị std dùng khi normalize

    Returns
    -------
    np.ndarray cùng shape với arr, đã nhân lại scale gốc.
    """
    return np.asarray(arr, dtype=np.float32) * std + mean
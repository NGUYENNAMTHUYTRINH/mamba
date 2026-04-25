"""
pipeline_lstm.py
----------------
LSTM model definition, training loop và pipeline chính cho Streamlit.
"""

from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import time
from datetime import datetime


import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from Utils import format_time_utc_strings, normalize_locations


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SequenceDataset(Dataset):
    """Dataset cho LSTM time-series forecasting."""

    def __init__(self, X: np.ndarray, loc_ids: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.loc_ids = torch.from_numpy(loc_ids).long()
        self.y = torch.from_numpy(y).float()

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.loc_ids[idx], self.y[idx]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class LSTMForecaster(nn.Module):
    """LSTM model cho time-series forecasting."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        horizon: int = 1,
        num_locations: int = 1,
        embed_dim: int = 8,
    ):
        super().__init__()
        self.use_embedding = num_locations > 1
        self.embed_dim = embed_dim
        if self.use_embedding:
            self.location_emb = nn.Embedding(num_locations, embed_dim)
            lstm_input_size = input_size + embed_dim
        else:
            lstm_input_size = input_size

        self.lstm = nn.LSTM(
            lstm_input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, horizon)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(
        self, x: torch.Tensor, loc_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        """x: (B, T, input_size) → output: (B, horizon)"""
        if self.use_embedding:
            if loc_ids is None:
                raise ValueError("loc_ids required when location embedding is enabled")
            loc_vec = self.location_emb(loc_ids)  # (B, E)
            loc_vec = loc_vec.unsqueeze(1).expand(-1, x.size(1), -1)  # (B, T, E)
            x = torch.cat([x, loc_vec], dim=-1)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_lstm_windows(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    lookback: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Tạo sliding windows cho LSTM từ time-series data."""
    features = df[feature_cols].values.astype(np.float32)
    target = df[target_col].values.astype(np.float32)

    X, y = [], []
    for i in range(len(features) - lookback - horizon + 1):
        X.append(features[i : i + lookback])
        y.append(target[i + lookback : i + lookback + horizon])

    if not X:
        return (
            np.array([]).reshape(0, lookback, len(feature_cols)),
            np.array([]).reshape(0, horizon),
        )
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


@torch.no_grad()
def evaluate_lstm(
    model: LSTMForecaster,
    loader: DataLoader,
    criterion,
    device,
    y_mean: float,
    y_std: float,
) -> dict:
    """Evaluate LSTM model trên một DataLoader."""
    model.eval()
    total_loss = 0.0
    preds, targets, loc_ids_all = [], [], []

    for xb, loc_ids, yb in loader:
        xb, loc_ids, yb = xb.to(device), loc_ids.to(device), yb.to(device)
        out = model(xb, loc_ids)
        loss = criterion(out, yb)

        total_loss += loss.item() * yb.size(0)
        preds.append(out.detach().cpu().numpy())
        targets.append(yb.detach().cpu().numpy())
        loc_ids_all.append(loc_ids.detach().cpu().numpy())

    preds_arr = np.concatenate(preds, axis=0) * y_std + y_mean
    targets_arr = np.concatenate(targets, axis=0) * y_std + y_mean

    mse = mean_squared_error(targets_arr, preds_arr)
    return {
        "loss": total_loss / len(loader.dataset),
        "mae": float(mean_absolute_error(targets_arr, preds_arr)),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(targets_arr.flatten(), preds_arr.flatten())),
        "preds": preds_arr,
        "targets": targets_arr,
        "loc_ids": (
            np.concatenate(loc_ids_all, axis=0)
            if loc_ids_all
            else np.array([], dtype=np.int64)
        ),
    }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_lstm_pipeline(
    df: pd.DataFrame,
    selected_locations: list[str],
    target_col: str,
    feature_cols: list[str],
    lookback: int,
    horizon: int,
    epochs: int,
    batch_size: int,
    lr: float,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    seed: int,
    use_gpu: bool,
    run_dir: str | None = None,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Chạy toàn bộ LSTM pipeline: train, val, test, forecast."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    selected_locations = normalize_locations(selected_locations)
    if not selected_locations:
        raise ValueError("LSTM yêu cầu chọn ít nhất 1 location")

    # --- Lọc & chuẩn bị data ---
    df_loc = df.loc[df["location_key"].astype(str).isin(selected_locations)].copy()
    if df_loc.empty:
        raise ValueError(f"No data found for selected locations: {selected_locations}")

    df_loc["ts_utc"] = pd.to_datetime(df_loc["ts_utc"], utc=True, errors="coerce")
    df_loc = (
        df_loc.dropna(subset=["ts_utc", "location_key"])
        .sort_values(["location_key", "ts_utc"])
        .reset_index(drop=True)
    )
    df_loc = df_loc.dropna(subset=feature_cols + [target_col])
    if len(df_loc) < lookback + horizon:
        raise ValueError(f"Not enough data for lookback={lookback} and horizon={horizon}")

    # --- Normalize ---
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()
    df_scaled = df_loc.copy()
    df_scaled[feature_cols] = X_scaler.fit_transform(df_loc[feature_cols])
    df_scaled[target_col] = y_scaler.fit_transform(df_loc[[target_col]]).flatten()

    # --- Tạo sequences per location ---
    loc_sorted = sorted(df_scaled["location_key"].astype(str).unique().tolist())
    loc_to_id = {loc: i for i, loc in enumerate(loc_sorted)}

    X_parts, y_parts, lid_parts, loc_name_parts, ts_parts = [], [], [], [], []
    for loc_name, g in df_scaled.groupby(
        df_scaled["location_key"].astype(str), sort=False
    ):
        g = g.sort_values("ts_utc").reset_index(drop=True)
        X_loc, y_loc = make_lstm_windows(g, feature_cols, target_col, lookback, horizon)
        if len(X_loc) == 0:
            continue
        X_parts.append(X_loc)
        y_parts.append(y_loc)
        lid_parts.append(
            np.full(len(X_loc), loc_to_id[str(loc_name)], dtype=np.int64)
        )
        loc_name_parts.append(np.full(len(X_loc), str(loc_name), dtype=object))
        ts_loc = g["ts_utc"].to_numpy(dtype="datetime64[ns]")
        ts_parts.append(
            ts_loc[lookback + horizon - 1 : lookback + horizon - 1 + len(X_loc)]
        )

    if not X_parts:
        raise ValueError("Could not create sequences from data")

    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    loc_ids = np.concatenate(lid_parts, axis=0)
    loc_names = np.concatenate(loc_name_parts, axis=0)
    sample_ts = np.concatenate(ts_parts, axis=0)

    # --- Split 70/10/20 theo timeline ---
    order = np.argsort(sample_ts)
    X, y, loc_ids, loc_names = X[order], y[order], loc_ids[order], loc_names[order]
    n = len(X)
    train_end = int(0.7 * n)
    val_end = int(0.8 * n)

    train_ds = SequenceDataset(X[:train_end], loc_ids[:train_end], y[:train_end])
    val_ds = SequenceDataset(
        X[train_end:val_end], loc_ids[train_end:val_end], y[train_end:val_end]
    )
    test_ds = SequenceDataset(X[val_end:], loc_ids[val_end:], y[val_end:])
    loc_name_test = loc_names[val_end:]
    test_time = sample_ts[order][val_end:]

    pin_memory = use_gpu and torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)

    device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")

    model = LSTMForecaster(
        input_size=len(feature_cols),
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        horizon=horizon,
        num_locations=len(loc_to_id),
        embed_dim=8,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    best_state = None
    history: list[dict] = []

    prog = st.progress(0)
    log_box = st.empty()
    log_lines: list[str] = []

    start_all = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        epoch_start = time.time()

        for xb, lid, yb in train_loader:
            xb, lid, yb = xb.to(device), lid.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb, lid)
            loss = criterion(out, yb)
            if torch.isfinite(loss):
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * yb.size(0)

        train_loss = running_loss / len(train_ds)
        val_metrics = evaluate_lstm(
            model, val_loader, criterion, device,
            y_scaler.mean_[0], y_scaler.scale_[0],
        )
        epoch_sec = time.time() - epoch_start

        line = (
            f"Epoch {epoch}/{epochs} | train_loss={train_loss:.6f} | "
            f"val_loss={val_metrics['loss']:.6f} | val_mae={val_metrics['mae']:.4f} | "
            f"val_rmse={val_metrics['rmse']:.4f} | val_r2={val_metrics['r2']:.4f} | "
            f"sec={epoch_sec:.1f}"
        )
        log_lines.append(line)
        log_box.code("\n".join(log_lines[-20:]))
        prog.progress(epoch / epochs)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "val_mae": val_metrics["mae"],
                "val_rmse": val_metrics["rmse"],
                "val_r2": val_metrics["r2"],
                "train_sec": epoch_sec,
            }
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("No valid checkpoint found during training.")

    model.load_state_dict(best_state)
    model.to(device)
    val_metrics = evaluate_lstm(model, val_loader, criterion, device, y_scaler.mean_[0], y_scaler.scale_[0])
    test_metrics = evaluate_lstm(model, test_loader, criterion, device, y_scaler.mean_[0], y_scaler.scale_[0])

    # --- Lưu artifacts ---
    if run_dir is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        lstm_run_dir = os.path.join("outputs", "lstm_runs", run_id)
    else:
        lstm_run_dir = run_dir
    os.makedirs(lstm_run_dir, exist_ok=True)

    model_path = os.path.join(lstm_run_dir, "best_lstm.pt")
    metrics_path = os.path.join(lstm_run_dir, "metrics_history.csv")
    pred_path = os.path.join(lstm_run_dir, "future_24h_predictions.csv")

    torch.save(
        {
            "model_state": model.state_dict(),
            "input_size": len(feature_cols),
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
            "horizon": horizon,
            "lookback": lookback,
            "feature_cols": feature_cols,
            "target_col": target_col,
            "X_scaler": X_scaler,
            "y_scaler": y_scaler,
        },
        model_path,
    )

    pd.DataFrame(history).to_csv(metrics_path, index=False)

    pred_df = pd.DataFrame(
        {
            "time": format_time_utc_strings(pd.Series(test_time)),
            "location": loc_name_test,
            "predicted": test_metrics["preds"].flatten(),
        }
    )
    pred_df.to_csv(pred_path, index=False)

    summary = {
        "model": "lstm",
        "device": str(device),
        "lookback": lookback,
        "horizon": horizon,
        "num_locations": len(loc_to_id),
        "selected_locations": ",".join(selected_locations),
        "val_mae": val_metrics["mae"],
        "val_rmse": val_metrics["rmse"],
        "val_r2": val_metrics["r2"],
        "test_mae": test_metrics["mae"],
        "test_rmse": test_metrics["rmse"],
        "test_r2": test_metrics["r2"],
        "train_only_sec": float(pd.DataFrame(history)["train_sec"].sum()) if history else 0.0,
        "run_sec": time.time() - start_all,
        "model_path": model_path,
        "metrics_path": metrics_path,
        "pred_path": pred_path,
    }

    return summary, pd.DataFrame(history), pred_df
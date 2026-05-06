"""
pipeline_lstm.py
----------------
LSTM model definition, training loop và pipeline chính cho Streamlit.

Cấu trúc train + predict được căn chỉnh 1:1 với Pipeline_mamba.py:
  - epoch_sec = train forward/backward + val evaluation (giống Mamba dòng 239-241)
  - Sau train: load best_state → eval val + test (giống Mamba dòng 271-276)
  - Forecast 24h: autoregressive rolling-window per location (giống Mamba dòng 318-365)
  - Lưu artifacts: model .pt, metrics_history.csv, future_24h_predictions.csv
  - summary keys giống Mamba: train_only_sec, eval_sec, forecast_sec, io_sec, run_sec
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
from torch.utils.data import DataLoader, Dataset

from Utils import (
    build_future_24h_frame,
    format_time_utc_strings,
    normalize_locations,
)


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
        embed_dim: int = 32,
    ):
        super().__init__()
        self.use_embedding = num_locations > 1
        self.embed_dim = embed_dim
        if self.use_embedding:
            self.location_emb = nn.Embedding(num_locations, embed_dim)
            lstm_input_size = input_size + embed_dim
        else:
            lstm_input_size = input_size

        self.input_proj = nn.Sequential(
            nn.Linear(lstm_input_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
        )

        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.output_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, horizon),
        )

        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(
        self, x: torch.Tensor, loc_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        """x: (B, T, input_size) → output: (B, horizon)"""
        if self.use_embedding:
            if loc_ids is None:
                raise ValueError("loc_ids required when location embedding is enabled")
            loc_vec = self.location_emb(loc_ids)           # (B, E)
            loc_vec = loc_vec.unsqueeze(1).expand(-1, x.size(1), -1)  # (B, T, E)
            x = torch.cat([x, loc_vec], dim=-1)
        x = self.input_proj(x)        # (B, T, hidden_size)
        out, _ = self.lstm(x)         # (B, T, hidden_size)
        return self.output_proj(out[:, -1, :])  # (B, horizon)


# ---------------------------------------------------------------------------
# Evaluate helper  (cấu trúc giống Mamba evaluate())
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, criterion, device, amp_enabled: bool,
             y_mean: float, y_std: float) -> dict:
    """Evaluate LSTM trên một DataLoader — cùng chữ ký với Mamba evaluate()."""
    model.eval()
    total_loss = 0.0
    preds, targets = [], []
    preds_norm, targets_norm = [], []

    for xb, loc_ids, yb in loader:
        xb      = xb.to(device, non_blocking=True)
        loc_ids = loc_ids.to(device, non_blocking=True)
        yb      = yb.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            out  = model(xb, loc_ids)
            loss = criterion(out, yb)

        total_loss += loss.item() * yb.size(0)
        preds_norm.append(out.detach().float().cpu().numpy())
        targets_norm.append(yb.detach().float().cpu().numpy())
        preds.append(out.detach().float().cpu().numpy())
        targets.append(yb.detach().float().cpu().numpy())

    preds_arr   = np.concatenate(preds,   axis=0) * y_std + y_mean
    targets_arr = np.concatenate(targets, axis=0) * y_std + y_mean

    preds_norm_arr = np.concatenate(preds_norm, axis=0)
    targets_norm_arr = np.concatenate(targets_norm, axis=0)

    mse_norm = mean_squared_error(targets_norm_arr, preds_norm_arr)
    mae_norm = mean_absolute_error(targets_norm_arr, preds_norm_arr)
    rmse_norm = float(np.sqrt(mse_norm))

    mse = mean_squared_error(targets_arr, preds_arr)
    return {
        "loss": total_loss / len(loader.dataset),
        "mae":  float(mean_absolute_error(targets_arr, preds_arr)),
        "rmse": float(np.sqrt(mse)),
        "r2":   float(r2_score(targets_arr, preds_arr)),
        "mae_norm": float(mae_norm),
        "rmse_norm": float(rmse_norm),
        "preds":   preds_arr,
        "targets": targets_arr,
    }


# ---------------------------------------------------------------------------
# Windowing  (numpy stride_tricks — nhanh hơn Python loop ~20x)
# ---------------------------------------------------------------------------

def _make_windows(features: np.ndarray, target: np.ndarray,
                  lookback: int, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    n = len(features)
    n_win = n - lookback - horizon + 1
    if n_win <= 0:
        n_feat = features.shape[1] if features.ndim > 1 else 1
        return (np.empty((0, lookback, n_feat), dtype=np.float32),
                np.empty((0, horizon),          dtype=np.float32))

    row_s, feat_s = features.strides
    X = np.lib.stride_tricks.as_strided(
        features,
        shape=(n_win, lookback, features.shape[1]),
        strides=(row_s, row_s, feat_s),
    ).copy().astype(np.float32)

    idx = np.arange(n_win)[:, None] + lookback + np.arange(horizon)[None, :]
    y   = target[idx].astype(np.float32)
    return X, y


# ---------------------------------------------------------------------------
# Pipeline  (cấu trúc 1:1 với Pipeline_mamba.py)
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
    loss_name: str,
    seed: int,
    num_workers: int,
    use_gpu: bool,
    log_interval: int,
    grad_accum_steps: int,
    max_grad_norm: float,
    run_dir: str | None = None,
    forecast_file_name: str = "future_24h_predictions.csv",
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Train LSTM và trả về (summary, history_df, future_df) — giống Mamba train_pipeline."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ── 1. Lọc & chuẩn bị data ───────────────────────────────────────────────
    selected_locations = normalize_locations(selected_locations)
    if not selected_locations:
        raise ValueError("LSTM yêu cầu chọn ít nhất 1 location.")

    work_df = df.loc[df["location_key"].astype(str).isin(selected_locations)].copy()
    if work_df.empty:
        raise ValueError(f"Không có dữ liệu cho locations: {selected_locations}")

    work_df["ts_utc"] = pd.to_datetime(work_df["ts_utc"], utc=True, errors="coerce")
    work_df = (
        work_df.dropna(subset=["ts_utc", "location_key"] + [target_col] + feature_cols)
        .sort_values(["location_key", "ts_utc"])
        .reset_index(drop=True)
    )
    if len(work_df) < lookback + horizon:
        raise ValueError(f"Không đủ dữ liệu cho lookback={lookback}, horizon={horizon}")

    # ── 2. Tạo sliding windows per location ──────────────────────────────────
    loc_sorted = sorted(work_df["location_key"].astype(str).unique().tolist())
    loc_to_id  = {loc: i for i, loc in enumerate(loc_sorted)}

    X_parts, y_parts, lid_parts, ts_parts = [], [], [], []
    for loc_name, g in work_df.groupby(work_df["location_key"].astype(str), sort=False):
        g = g.sort_values("ts_utc").reset_index(drop=True)
        feats  = g[feature_cols].values.astype(np.float32)
        tgt    = g[target_col].values.astype(np.float32)
        X_loc, y_loc = _make_windows(feats, tgt, lookback, horizon)
        if len(X_loc) == 0:
            continue
        X_parts.append(X_loc)
        y_parts.append(y_loc)
        lid_parts.append(np.full(len(X_loc), loc_to_id[str(loc_name)], dtype=np.int64))
        ts_arr = g["ts_utc"].to_numpy(dtype="datetime64[ns]")
        ts_parts.append(ts_arr[lookback + horizon - 1 : lookback + horizon - 1 + len(X_loc)])

    if not X_parts:
        raise ValueError("Không tạo được sequences từ dữ liệu.")

    X        = np.concatenate(X_parts,   axis=0)
    y_raw    = np.concatenate(y_parts,   axis=0)
    loc_ids  = np.concatenate(lid_parts, axis=0)
    sample_ts = np.concatenate(ts_parts, axis=0)

    # ── 3. Split 70/10/20 theo timeline (giống Mamba split_data_by_timeline) ─
    order = np.argsort(sample_ts)
    X, y_raw, loc_ids, sample_ts = (
        X[order], y_raw[order], loc_ids[order], sample_ts[order]
    )
    n         = len(X)
    train_end = int(0.7 * n)
    val_end   = int(0.8 * n)

    # ── 4. Normalize (giống Mamba: fit trên train, apply cho cả 3) ────────────
    x_mean = X[:train_end].mean(axis=(0, 1), keepdims=True)   # (1, 1, n_feat)
    x_std  = X[:train_end].std(axis=(0, 1),  keepdims=True)   # (1, 1, n_feat)
    x_std  = np.where(x_std < 1e-6, 1.0, x_std)
    X      = (X - x_mean) / x_std

    # Squeeze về (n_feat,) để dùng broadcast với rolling_window shape (lookback, n_feat)
    x_mean_1d = x_mean.squeeze()   # (n_feat,)
    x_std_1d  = x_std.squeeze()    # (n_feat,)

    y_mean = float(y_raw[:train_end].mean())
    y_std  = float(y_raw[:train_end].std())
    if y_std < 1e-6:
        y_std = 1.0
    y_norm = (y_raw - y_mean) / y_std

    train_ds = SequenceDataset(X[:train_end],        loc_ids[:train_end],        y_norm[:train_end])
    val_ds   = SequenceDataset(X[train_end:val_end], loc_ids[train_end:val_end], y_norm[train_end:val_end])
    test_ds  = SequenceDataset(X[val_end:],          loc_ids[val_end:],          y_norm[val_end:])

    # ── 5. Device & AMP (giống Mamba) ────────────────────────────────────────
    device     = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")
    pin_memory = device.type == "cuda"
    amp_enabled = device.type == "cuda"

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    else:
        cpu_threads = max(1, (os.cpu_count() or 2) - 1)
        torch.set_num_threads(cpu_threads)

    # ── 6. DataLoader (giống Mamba) ───────────────────────────────────────────
    loader_kwargs: dict = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(train_ds, shuffle=(device.type == "cuda"), **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False,                   **loader_kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False,                   **loader_kwargs)

    # ── 7. Model, loss, optimizer, scaler (giống Mamba) ──────────────────────
    model = LSTMForecaster(
        input_size=len(feature_cols),
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        horizon=horizon,
        num_locations=len(loc_to_id),
        embed_dim=32,
    ).to(device)

    criterion = nn.HuberLoss(delta=1.0) if loss_name == "huber" else nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler    = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    best_val_loss = float("inf")
    best_state    = None
    history: list[dict] = []

    total_steps = epochs * len(train_loader)
    global_step = 0
    prog    = st.progress(0)
    log_box = st.empty()
    log_lines: list[str] = []

    # ── 8. Training loop (1:1 với Mamba) ─────────────────────────────────────
    start_all = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        epoch_start  = time.time()
        optimizer.zero_grad(set_to_none=True)

        for step, (xb, loc_batch, yb) in enumerate(train_loader, start=1):
            xb        = xb.to(device, non_blocking=pin_memory)
            loc_batch = loc_batch.to(device, non_blocking=pin_memory)
            yb        = yb.to(device, non_blocking=pin_memory)

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                out  = model(xb, loc_batch)
                loss = criterion(out, yb)

            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                continue

            loss_for_backward = loss / grad_accum_steps
            if amp_enabled:
                scaler.scale(loss_for_backward).backward()
            else:
                loss_for_backward.backward()

            if step % grad_accum_steps == 0 or step == len(train_loader):
                if amp_enabled:
                    scaler.unscale_(optimizer)
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                if amp_enabled:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item() * yb.size(0)
            global_step  += 1

            if total_steps > 0 and (step % 20 == 0 or step == len(train_loader)):
                prog.progress(min(global_step / total_steps, 1.0))

            if log_interval > 0 and (step % log_interval == 0 or step == len(train_loader)):
                avg_loss = running_loss / max(step * yb.size(0), 1)
                line = (
                    f"Epoch {epoch}/{epochs} | step {step}/{len(train_loader)} | "
                    f"batch_loss={loss.item():.6f} | running_avg={avg_loss:.6f}"
                )
                log_lines.append(line)
                log_box.code("\n".join(log_lines[-20:]))

        # epoch_sec = train + val (giống Mamba dòng 239-241)
        train_loss   = running_loss / len(train_loader.dataset)
        val_metrics  = evaluate(model, val_loader, criterion, device, amp_enabled, y_mean, y_std)
        epoch_sec    = time.time() - epoch_start

        epoch_line = (
            f"Epoch {epoch}/{epochs} done | train_loss={train_loss:.6f} | "
            f"val_loss={val_metrics['loss']:.6f} | mae={val_metrics.get('mae_norm', float('nan')):.4f} | "
            f"rmse={val_metrics.get('rmse_norm', float('nan')):.4f} | val_r2={val_metrics['r2']:.4f} | "
            f"sec={epoch_sec:.1f}"
        )
        log_lines.append(epoch_line)
        log_box.code("\n".join(log_lines[-20:]))

        history.append({
            "epoch":      epoch,
            "train_loss": train_loss,
            "val_loss":   val_metrics["loss"],
            "mae":        val_metrics.get("mae_norm"),
            "rmse":       val_metrics.get("rmse_norm"),
            "val_r2":     val_metrics["r2"],
            "train_sec":  epoch_sec,        # train + val — giống Mamba
        })

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Không có checkpoint hợp lệ trong quá trình train.")

    # ── 9. Load best → eval val + test (giống Mamba dòng 271-276) ────────────
    model.load_state_dict(best_state)
    model.to(device)
    eval_start   = time.time()
    val_metrics  = evaluate(model, val_loader,  criterion, device, amp_enabled, y_mean, y_std)
    test_metrics = evaluate(model, test_loader, criterion, device, amp_enabled, y_mean, y_std)
    eval_sec     = time.time() - eval_start


    # ── 10. Forecast 24h autoregressive rolling-window (giống Mamba dòng 290-365)
    # Dùng toàn bộ work_df làm base để lấy 24 bước lịch sử cuối

    base_df = work_df.copy()
    future_df = build_future_24h_frame(base_df, feature_cols=feature_cols, target_col=target_col)

    for col in feature_cols:
        future_df[col] = pd.to_numeric(future_df[col], errors="coerce")
        fill_val = base_df[col].median() if col in base_df.columns else 0.0
        if pd.isna(fill_val):
            fill_val = 0.0
        future_df[col] = future_df[col].fillna(fill_val)

    forecast_start  = time.time()
    preds_rows: list[dict] = []
    model.eval()
    infer_x, infer_loc, infer_meta = [], [], []

    with torch.inference_mode():
        for loc in sorted(future_df["location_key"].astype(str).unique().tolist()):
            if loc not in loc_to_id:
                continue
            loc_hist = (
                base_df.loc[base_df["location_key"].astype(str) == loc]
                .copy()
                .assign(ts_utc=lambda d: pd.to_datetime(d["ts_utc"], utc=True, errors="coerce"))
                .dropna(subset=["ts_utc"])
                .sort_values("ts_utc")
            )
            if len(loc_hist) < lookback:
                continue

            rolling_window = loc_hist[feature_cols].tail(lookback).to_numpy(dtype=np.float32)
            loc_future = (
                future_df.loc[future_df["location_key"].astype(str) == loc]
                .copy()
                .assign(ts_utc=lambda d: pd.to_datetime(d["ts_utc"], utc=True, errors="coerce"))
                .sort_values("ts_utc")
            )

            for _, row in loc_future.iterrows():
                x_norm = (rolling_window - x_mean_1d) / x_std_1d
                infer_x.append(x_norm.astype(np.float32, copy=False))
                infer_loc.append(int(loc_to_id[loc]))
                infer_meta.append((row["ts_utc"], loc))
                next_feats = row[feature_cols].to_numpy(dtype=np.float32).reshape(1, -1)
                rolling_window = np.concatenate([rolling_window[1:], next_feats], axis=0)

        if infer_x:
            x_all   = torch.from_numpy(np.stack(infer_x, axis=0)).to(device, non_blocking=pin_memory)
            loc_all = torch.tensor(infer_loc, dtype=torch.long, device=device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                pred_norm_all = model(x_all, loc_all).detach().float().cpu().numpy()

            # flatten() để chuyển (N,1) → (N,) trước khi gọi float() — fix numpy>=2.0
            pred_all = (pred_norm_all * y_std + y_mean).flatten()
            for (ts_val, loc_val), pred_val in zip(infer_meta, pred_all):
                preds_rows.append({"time": ts_val, "location": loc_val, "predicted": float(pred_val)})

    forecast_sec = time.time() - forecast_start

    if not preds_rows:
        raise RuntimeError("Không tạo được dự báo 24h cho LSTM.")

    future_out = (
        pd.DataFrame(preds_rows)
        .assign(time=lambda d: format_time_utc_strings(d["time"]))
        [["time", "location", "predicted"]]
        .sort_values(["location", "time"])
        .reset_index(drop=True)
    )

    # ── 11. Lưu artifacts (giống Mamba) ──────────────────────────────────────
    if run_dir is None:
        run_id  = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join("outputs", "lstm_runs", run_id)
    else:
        out_dir = run_dir
    os.makedirs(out_dir, exist_ok=True)

    model_path       = os.path.join(out_dir, "best_lstm.pt")
    metrics_path     = os.path.join(out_dir, "metrics_history.csv")
    future_pred_path = os.path.join(out_dir, forecast_file_name)

    io_start = time.time()
    torch.save(model.state_dict(), model_path)
    pd.DataFrame(history).to_csv(metrics_path, index=False)
    future_out.to_csv(future_pred_path, index=False)
    io_sec = time.time() - io_start

    # ── 12. Summary (keys giống Mamba) ───────────────────────────────────────
    summary = {
        "model":        "lstm",
        "device":       str(device),
        "n_rows_used":  len(y_raw),
        "split_train":  len(train_ds),
        "split_val":    len(val_ds),
        "split_test":   len(test_ds),
        "val_loss":     val_metrics["loss"],
        "val_r2":       val_metrics["r2"],
        "val_mae_norm": val_metrics.get("mae_norm"),
        "val_rmse_norm": val_metrics.get("rmse_norm"),
        "test_loss":    test_metrics["loss"],
        "test_mae":     test_metrics["mae"],
        "test_rmse":    test_metrics["rmse"],
        "test_r2":      test_metrics["r2"],
        "test_mae_norm": test_metrics.get("mae_norm"),
        "test_rmse_norm": test_metrics.get("rmse_norm"),
        "model_path":         model_path,
        "metrics_path":       metrics_path,
        "future_pred_path":   future_pred_path,
        "future_rows":        len(future_out),
        "future_locations":   int(future_out["location"].nunique()),
        "per_location_files": [],
        # timing — giống Mamba summary
        "train_only_sec": float(pd.DataFrame(history)["train_sec"].sum()) if history else 0.0,
        "eval_sec":       float(eval_sec),
        "forecast_sec":   float(forecast_sec),
        "io_sec":         float(io_sec),
        "run_sec":        time.time() - start_all,
    }

    return summary, pd.DataFrame(history), future_out
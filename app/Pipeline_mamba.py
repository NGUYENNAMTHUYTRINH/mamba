"""
pipeline_mamba.py
-----------------
Mamba model (TabularMambaRegressor), training loop, evaluate, và train_pipeline.
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
from mamba_ssm import Mamba
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, Dataset

from Utils import (
    build_future_24h_frame,
    format_time_utc_strings,
    load_train_module,
    normalize_locations,
    split_data_by_timeline,
)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TabularDataset(Dataset):
    """Dataset cho Mamba tabular forecasting."""

    def __init__(self, x: np.ndarray, loc_ids: np.ndarray, y: np.ndarray):
        self.x = torch.from_numpy(x).float()
        self.loc_ids = torch.from_numpy(loc_ids).long()
        self.y = torch.from_numpy(y).float()

    def __len__(self) -> int:
        return self.y.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.loc_ids[idx], self.y[idx]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class TabularMambaRegressor(nn.Module):
    """Mamba-based tabular regressor với location embedding."""

    def __init__(
        self,
        num_features: int,
        num_locations: int,
        d_model: int = 64,
        n_layers: int = 2,
    ):
        super().__init__()
        self.scalar_proj = nn.Linear(1, d_model)
        self.location_emb = nn.Embedding(num_locations, d_model)
        self.layers = nn.ModuleList(
            [
                Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2, use_fast_path=False)
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.num_features = num_features

    def forward(self, x_num: torch.Tensor, loc_ids: torch.Tensor) -> torch.Tensor:
        # x_num: (B, F)
        x = self.scalar_proj(x_num.unsqueeze(-1))  # (B, F, d_model)
        loc_token = self.location_emb(loc_ids).unsqueeze(1)  # (B, 1, d_model)
        x = torch.cat([loc_token, x], dim=1)  # (B, F+1, d_model)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.head(x.mean(dim=1)).squeeze(-1)


# ---------------------------------------------------------------------------
# Evaluate helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, criterion, device, y_mean: float, y_std: float) -> dict:
    """Evaluate Mamba model trên một DataLoader."""
    model.eval()
    total_loss = 0.0
    preds, targets = [], []

    for xb, loc_ids, yb in loader:
        xb, loc_ids, yb = xb.to(device), loc_ids.to(device), yb.to(device)
        out = model(xb, loc_ids)
        loss = criterion(out, yb)
        total_loss += loss.item() * yb.size(0)
        preds.append(out.detach().cpu().numpy())
        targets.append(yb.detach().cpu().numpy())

    preds_arr = np.concatenate(preds, axis=0) * y_std + y_mean
    targets_arr = np.concatenate(targets, axis=0) * y_std + y_mean

    mse = mean_squared_error(targets_arr, preds_arr)
    return {
        "loss": total_loss / len(loader.dataset),
        "mae": float(mean_absolute_error(targets_arr, preds_arr)),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(targets_arr, preds_arr)),
        "preds": preds_arr,
        "targets": targets_arr,
    }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def train_pipeline(
    df: pd.DataFrame,
    forecast_base_df: pd.DataFrame | None,
    selected_locations: list[str],
    target_col: str,
    feature_cols: list[str],
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    d_model: int,
    n_layers: int,
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
    """Train Mamba sequence model và trả về (summary, history_df, future_df)."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    selected_locations = normalize_locations(selected_locations)
    if not selected_locations:
        raise ValueError("Cần chọn ít nhất 1 location để train Mamba.")

    work_df = df.copy()
    if "location_key" not in work_df.columns or "ts_utc" not in work_df.columns:
        raise ValueError("Dataset train Mamba cần có cột 'location_key' và 'ts_utc'.")

    work_df = work_df.loc[
        work_df["location_key"].astype(str).isin([str(x) for x in selected_locations])
    ].copy()
    if work_df.empty:
        raise ValueError("Không có dữ liệu train cho các location đã chọn.")

    # --- Load helper từ scripts/train_mamba_aqi.py ---
    mod = load_train_module()
    if mod is None or not hasattr(mod, "build_time_series_samples"):
        raise RuntimeError(
            "Không load được module mamba/scripts/train_mamba_aqi.py để chạy Mamba sequence."
        )

    window_size = 24
    horizon = 1
    x_seq, loc_ids, y, y_ts, num_locations, ts_feature_cols = mod.build_time_series_samples(
        df=work_df, target_col=target_col, window_size=window_size, horizon=horizon
    )

    train_split, val_split, test_split = mod.split_data_by_timeline(x_seq, loc_ids, y, y_ts)

    # --- Normalize ---
    x_mean = train_split.x_seq.mean(axis=(0, 1), keepdims=True)
    x_std = train_split.x_seq.std(axis=(0, 1), keepdims=True)
    x_std = np.where(x_std < 1e-6, 1.0, x_std)
    for s in [train_split, val_split, test_split]:
        s.x_seq = (s.x_seq - x_mean) / x_std

    y_mean = float(train_split.y.mean())
    y_std = float(train_split.y.std())
    if y_std < 1e-6:
        y_std = 1.0
    for s in [train_split, val_split, test_split]:
        s.y = (s.y - y_mean) / y_std

    train_ds = mod.AQIDataset(train_split)
    val_ds = mod.AQIDataset(val_split)
    test_ds = mod.AQIDataset(test_split)

    device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")
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

    loader_kwargs: dict = {"batch_size": batch_size, "num_workers": num_workers, "pin_memory": pin_memory}
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(train_ds, shuffle=(device.type == "cuda"), **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    model = mod.TimeSeriesMambaRegressor(
        num_features=train_split.x_seq.shape[-1],
        num_locations=num_locations,
        d_model=d_model,
        n_layers=n_layers,
    ).to(device)

    criterion = nn.HuberLoss(delta=1.0) if loss_name == "huber" else nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = float("inf")
    best_state = None
    history: list[dict] = []
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    total_steps = epochs * len(train_loader)
    global_step = 0
    prog = st.progress(0)
    log_box = st.empty()
    log_lines: list[str] = []

    start_all = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        epoch_start = time.time()
        optimizer.zero_grad(set_to_none=True)

        for step, (xb, loc_batch, yb) in enumerate(train_loader, start=1):
            xb = xb.to(device, non_blocking=pin_memory)
            loc_batch = loc_batch.to(device, non_blocking=pin_memory)
            yb = yb.to(device, non_blocking=pin_memory)

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                out = model(xb, loc_batch)
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
            global_step += 1

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

        train_loss = running_loss / len(train_loader.dataset)
        val_metrics = mod.evaluate(model, val_loader, criterion, device, amp_enabled, y_mean, y_std)
        epoch_sec = time.time() - epoch_start

        epoch_line = (
            f"Epoch {epoch}/{epochs} done | train_loss={train_loss:.6f} | "
            f"val_loss={val_metrics['loss']:.6f} | val_mae={val_metrics['mae']:.4f} | "
            f"val_rmse={val_metrics['rmse']:.4f} | val_r2={val_metrics['r2']:.4f} | "
            f"sec={epoch_sec:.1f}"
        )
        log_lines.append(epoch_line)
        log_box.code("\n".join(log_lines[-20:]))

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
        raise RuntimeError("Không có checkpoint hợp lệ trong quá trình train.")

    model.load_state_dict(best_state)
    model.to(device)
    eval_start = time.time()
    val_metrics = mod.evaluate(model, val_loader, criterion, device, amp_enabled, y_mean, y_std)
    test_metrics = mod.evaluate(model, test_loader, criterion, device, amp_enabled, y_mean, y_std)
    eval_sec = time.time() - eval_start

    # --- Build loc_to_id mapping ---
    cleaned = work_df.copy()
    cleaned["_ts"] = pd.to_datetime(cleaned["ts_utc"], utc=True, errors="coerce")
    cleaned = cleaned.dropna(subset=["_ts", "location_key", target_col]).copy()
    cleaned["_loc_id"] = cleaned["location_key"].astype("category").cat.codes.astype(np.int64)
    loc_to_id = (
        cleaned.assign(_loc_key_str=cleaned["location_key"].astype(str))
        .drop_duplicates(subset=["_loc_key_str"])
        .set_index("_loc_key_str")["_loc_id"]
        .to_dict()
    )

    # --- Forecast 24h ---
    base_df = forecast_base_df if forecast_base_df is not None else cleaned.copy()
    if not isinstance(base_df, pd.DataFrame) or base_df.empty:
        raise ValueError("Không có dữ liệu test làm mốc để dự báo 24h tiếp theo.")

    base_df = base_df.loc[
        base_df["location_key"].astype(str).isin(list(loc_to_id.keys()))
    ].copy()
    if base_df.empty:
        raise ValueError("Test CSV không có location trùng với dữ liệu train đã chọn.")

    for col in ts_feature_cols:
        if col not in base_df.columns:
            base_df[col] = np.nan
        base_df[col] = pd.to_numeric(base_df[col], errors="coerce")
        fill_val = base_df[col].median()
        if pd.isna(fill_val):
            fill_val = 0.0
        base_df[col] = base_df[col].fillna(fill_val)

    future_df = build_future_24h_frame(base_df, feature_cols=ts_feature_cols, target_col=target_col)
    for col in ts_feature_cols:
        future_df[col] = pd.to_numeric(future_df[col], errors="coerce")
        fill_val = base_df[col].median() if col in base_df.columns else 0.0
        if pd.isna(fill_val):
            fill_val = 0.0
        future_df[col] = future_df[col].fillna(fill_val)

    # --- Batched inference ---
    forecast_start = time.time()
    preds_rows: list[dict] = []
    model.eval()
    infer_x, infer_loc, infer_meta = [], [], []
    x_mean_2d = x_mean.squeeze(0)
    x_std_2d = x_std.squeeze(0)

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
            if len(loc_hist) < window_size:
                continue

            rolling_window = loc_hist[ts_feature_cols].tail(window_size).to_numpy(dtype=np.float32)
            loc_future = (
                future_df.loc[future_df["location_key"].astype(str) == loc]
                .copy()
                .assign(ts_utc=lambda d: pd.to_datetime(d["ts_utc"], utc=True, errors="coerce"))
                .sort_values("ts_utc")
            )

            for _, row in loc_future.iterrows():
                x_norm = (rolling_window - x_mean_2d) / x_std_2d
                infer_x.append(x_norm.astype(np.float32, copy=False))
                infer_loc.append(int(loc_to_id[loc]))
                infer_meta.append((row["ts_utc"], loc))
                next_feats = row[ts_feature_cols].to_numpy(dtype=np.float32).reshape(1, -1)
                rolling_window = np.concatenate([rolling_window[1:], next_feats], axis=0)

        if infer_x:
            x_all = torch.from_numpy(np.stack(infer_x, axis=0)).to(device, non_blocking=pin_memory)
            loc_all = torch.tensor(infer_loc, dtype=torch.long, device=device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                pred_norm_all = model(x_all, loc_all).detach().float().cpu().numpy()

            pred_all = pred_norm_all * y_std + y_mean
            for (ts_val, loc_val), pred_val in zip(infer_meta, pred_all):
                preds_rows.append({"time": ts_val, "location": loc_val, "predicted": float(pred_val)})

    forecast_sec = time.time() - forecast_start

    if not preds_rows:
        raise RuntimeError("Không tạo được dự báo 24h cho Mamba sequence.")

    future_out = (
        pd.DataFrame(preds_rows)
        .assign(time=lambda d: format_time_utc_strings(d["time"]))
        [["time", "location", "predicted"]]
        .sort_values(["location", "time"])
        .reset_index(drop=True)
    )

    # --- Lưu artifacts ---
    if run_dir is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join("outputs", "streamlit_runs", run_id)
    else:
        out_dir = run_dir
    os.makedirs(out_dir, exist_ok=True)

    model_path = os.path.join(out_dir, "best_mamba.pt")
    metrics_path = os.path.join(out_dir, "metrics_history.csv")
    future_pred_path = os.path.join(out_dir, forecast_file_name)

    io_start = time.time()
    torch.save(model.state_dict(), model_path)
    pd.DataFrame(history).to_csv(metrics_path, index=False)
    future_out.to_csv(future_pred_path, index=False)
    io_sec = time.time() - io_start

    summary = {
        "device": str(device),
        "n_rows_used": len(y),
        "split_train": len(train_split.y),
        "split_val": len(val_split.y),
        "split_test": len(test_split.y),
        "feature_count_after_encode": train_split.x_seq.shape[-1],
        "encoded_features": ts_feature_cols,
        "val_loss": val_metrics["loss"],
        "val_mae": val_metrics["mae"],
        "val_rmse": val_metrics["rmse"],
        "val_r2": val_metrics["r2"],
        "test_loss": test_metrics["loss"],
        "test_mae": test_metrics["mae"],
        "test_rmse": test_metrics["rmse"],
        "test_r2": test_metrics["r2"],
        "model_path": model_path,
        "metrics_path": metrics_path,
        "future_pred_path": future_pred_path,
        "future_rows": len(future_out),
        "future_locations": int(future_out["location"].nunique()),
        "per_location_files": [],
        "train_only_sec": float(pd.DataFrame(history)["train_sec"].sum()) if history else 0.0,
        "eval_sec": float(eval_sec),
        "forecast_sec": float(forecast_sec),
        "io_sec": float(io_sec),
        "run_sec": time.time() - start_all,
    }

    return summary, pd.DataFrame(history), future_out
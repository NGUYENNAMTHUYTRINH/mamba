import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader

import importlib.util
from pathlib import Path


def _load_tft_class():
    """Load TFT class from Transformer/models/temporal_fusion_t."""
    try:
        transformer_root = Path(__file__).parent / "Transformer"
        if str(transformer_root) not in sys.path:
            sys.path.insert(0, str(transformer_root))
        from models.temporal_fusion_t.tft_model import TFT  # type: ignore

        return TFT
    except Exception:
        return None


def _load_train_module():
    """Dynamically load scripts/train_mamba_aqi.py and return the module.
    This avoids package import issues when running Streamlit.
    """
    try:
        mod_path = Path(__file__).parent / "scripts" / "train_mamba_aqi.py"
        spec = importlib.util.spec_from_file_location("train_mamba_aqi_for_streamlit", str(mod_path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    except Exception:
        return None


def sanitize_filename(text: str) -> str:
    safe = []
    for ch in str(text).strip().lower():
        if ch.isalnum() or ch in ["_", "-"]:
            safe.append(ch)
        else:
            safe.append("_")
    name = "".join(safe).strip("_")
    return name or "unknown_location"


def _predict_future_from_rows(
    model,
    combined_df: pd.DataFrame,
    future_df: pd.DataFrame,
    numeric_cols: list[str],
    window_size: int,
    horizon: int,
    device,
    y_mean: float,
    y_std: float,
    x_mean: np.ndarray,
    x_std: np.ndarray,
):
    combined_work = combined_df.copy()
    for col in numeric_cols:
        if col not in combined_work.columns:
            combined_work[col] = 0.0
        med = pd.to_numeric(combined_work[col], errors="coerce").median()
        if pd.isna(med):
            med = 0.0
        combined_work[col] = pd.to_numeric(combined_work[col], errors="coerce").fillna(float(med))

    preds = []
    combined_work = combined_work.reset_index(drop=True)
    future_ts = pd.to_datetime(future_df["ts_utc"], utc=True, errors="coerce")
    combined_ts = pd.to_datetime(combined_work["ts_utc"], utc=True, errors="coerce")

    # Use integer nanosecond timestamps for robust matching between timezone-aware values.
    future_ns = future_ts.astype("int64").to_numpy()
    combined_ns = combined_ts.astype("int64").to_numpy()

    for ts_ns in future_ns:
        idx_arr = np.where(combined_ns == ts_ns)[0]
        if len(idx_arr) == 0:
            preds.append(np.nan)
            continue

        target_idx = int(idx_arr[0])
        win_end = target_idx - horizon
        win_start = win_end - window_size + 1
        if win_start < 0 or win_end < 0:
            preds.append(np.nan)
            continue

        x_window = combined_work.loc[win_start:win_end, numeric_cols].to_numpy(dtype=np.float32)
        if x_window.shape[0] != window_size:
            preds.append(np.nan)
            continue

        x_window = (x_window - x_mean[0, 0, :]) / x_std[0, 0, :]
        x_tensor = torch.from_numpy(x_window[None, :, :]).float().to(device)
        loc_tensor = torch.zeros((1,), dtype=torch.long, device=device)
        with torch.no_grad():
            pred = model(x_tensor, loc_tensor).detach().cpu().numpy()[0]
        preds.append(float(pred * y_std + y_mean))

    return np.asarray(preds, dtype=np.float32)


def build_future_24h_frame(df_valid: pd.DataFrame, feature_cols: list[str], target_col: str) -> pd.DataFrame:
    if "ts_utc" not in df_valid.columns:
        raise ValueError("Cần có cột 'ts_utc' để dự báo 24h tiếp theo.")

    work = df_valid.copy()
    work["ts_utc"] = pd.to_datetime(work["ts_utc"], utc=True, errors="coerce")
    work = work.dropna(subset=["ts_utc"]).copy()

    future_rows = []
    if "location_key" in work.columns:
        groups = [
            (loc, work.loc[work["location_key"].astype(str) == loc].sort_values("ts_utc").copy())
            for loc in sorted(work["location_key"].dropna().astype(str).unique().tolist())
        ]
    else:
        groups = [(None, work.sort_values("ts_utc").copy())]

    for loc, g in groups:
        if g.empty:
            continue

        last_ts = g["ts_utc"].iloc[-1]
        # Forecast the NEXT CALENDAR DAY (00:00 -> 23:00) after the last observed test timestamp.
        next_day_start = last_ts.normalize() + pd.Timedelta(days=1)
        template = g.tail(24).copy()
        if len(template) < 24:
            template = pd.concat([template] * (24 // len(template) + 1), ignore_index=True).head(24)

        for h in range(24):
            src = template.iloc[h].copy()
            row = {col: src[col] for col in feature_cols if col in template.columns}
            if loc is not None:
                row["location_key"] = loc
            row["ts_utc"] = next_day_start + pd.Timedelta(hours=h)
            row[target_col] = np.nan
            future_rows.append(row)

    if not future_rows:
        raise ValueError("Không tạo được dữ liệu dự báo 24h.")

    return pd.DataFrame(future_rows)


@torch.no_grad()
def evaluate(model, loader, criterion, device, y_mean, y_std):
    model.eval()
    total_loss = 0.0
    preds = []
    targets = []

    for xb, loc_ids, yb in loader:
        xb = xb.to(device)
        loc_ids = loc_ids.to(device)
        yb = yb.to(device)

        out = model(xb, loc_ids)
        loss = criterion(out, yb)

        total_loss += loss.item() * yb.size(0)
        preds.append(out.detach().cpu().numpy())
        targets.append(yb.detach().cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    targets = np.concatenate(targets, axis=0)

    preds = preds * y_std + y_mean
    targets = targets * y_std + y_mean

    mse = mean_squared_error(targets, preds)
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(targets, preds))
    r2 = float(r2_score(targets, preds))

    return {
        "loss": total_loss / len(loader.dataset),
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "preds": preds,
        "targets": targets,
    }


@torch.no_grad()
def evaluate_tft(model, loader, criterion, device, y_mean, y_std):
    model.eval()
    total_loss = 0.0
    preds = []
    targets = []

    for xb, _, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        future_row = xb[:, -1:, :].clone()
        x_in = torch.cat([xb, future_row], dim=1)

        out, _, _ = model(x_in)
        # out: (B, horizon, output_size * num_quantiles), use first output for regression
        pred = out[:, -1, 0]
        loss = criterion(pred, yb)

        total_loss += loss.item() * yb.size(0)
        preds.append(pred.detach().cpu().numpy())
        targets.append(yb.detach().cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    targets = np.concatenate(targets, axis=0)

    preds = preds * y_std + y_mean
    targets = targets * y_std + y_mean

    mse = mean_squared_error(targets, preds)
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(targets, preds))
    r2 = float(r2_score(targets, preds))

    return {
        "loss": total_loss / len(loader.dataset),
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "preds": preds,
        "targets": targets,
    }


def _safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    denom = np.where(np.abs(y_true) < 1e-6, np.nan, np.abs(y_true))
    mape = np.nanmean(np.abs((y_true - y_pred) / denom)) * 100.0
    if np.isnan(mape):
        return float("nan")
    return float(mape)


def _predict_future_from_rows_tft(
    model,
    combined_df: pd.DataFrame,
    future_df: pd.DataFrame,
    numeric_cols: list[str],
    window_size: int,
    horizon: int,
    device,
    y_mean: float,
    y_std: float,
    x_mean: np.ndarray,
    x_std: np.ndarray,
):
    combined_work = combined_df.copy()
    for col in numeric_cols:
        if col not in combined_work.columns:
            combined_work[col] = 0.0
        med = pd.to_numeric(combined_work[col], errors="coerce").median()
        if pd.isna(med):
            med = 0.0
        combined_work[col] = pd.to_numeric(combined_work[col], errors="coerce").fillna(float(med))

    preds = []
    combined_work = combined_work.reset_index(drop=True)
    future_ts = pd.to_datetime(future_df["ts_utc"], utc=True, errors="coerce")
    combined_ts = pd.to_datetime(combined_work["ts_utc"], utc=True, errors="coerce")
    future_ns = future_ts.astype("int64").to_numpy()
    combined_ns = combined_ts.astype("int64").to_numpy()

    for ts_ns in future_ns:
        idx_arr = np.where(combined_ns == ts_ns)[0]
        if len(idx_arr) == 0:
            preds.append(np.nan)
            continue

        target_idx = int(idx_arr[0])
        win_end = target_idx - horizon
        win_start = win_end - window_size + 1
        if win_start < 0 or win_end < 0:
            preds.append(np.nan)
            continue

        x_window = combined_work.loc[win_start:win_end, numeric_cols].to_numpy(dtype=np.float32)
        if x_window.shape[0] != window_size:
            preds.append(np.nan)
            continue

        x_window = (x_window - x_mean[0, 0, :]) / x_std[0, 0, :]
        future_row = combined_work.loc[target_idx, numeric_cols].to_numpy(dtype=np.float32)
        future_row = (future_row - x_mean[0, 0, :]) / x_std[0, 0, :]
        model_input = np.concatenate([x_window, future_row[None, :]], axis=0)

        x_tensor = torch.from_numpy(model_input[None, :, :]).float().to(device)
        with torch.no_grad():
            out, _, _ = model(x_tensor)
            pred = out[:, -1, 0].detach().cpu().numpy()[0]
        preds.append(float(pred * y_std + y_mean))

    return np.asarray(preds, dtype=np.float32)


def train_pipeline(
    df: pd.DataFrame,
    forecast_base_df: pd.DataFrame | None,
    selected_locations: list[str],
    target_col: str,
    window_size: int,
    horizon: int,
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
    input_feature_cols: list[str] | None = None,
    include_target_history: bool = True,
    run_dir: str | None = None,
    forecast_file_name: str = "future_24h_predictions.csv",
):
    mod = _load_train_module()
    if mod is None:
        raise RuntimeError("Không thể load scripts/train_mamba_aqi.py")
    required = [
        "build_time_series_samples",
        "split_data_by_timeline",
        "standardize",
        "AQIDataset",
        "TimeSeriesMambaRegressorNoLoc",
    ]
    for name in required:
        if not hasattr(mod, name):
            raise RuntimeError(f"Thiếu helper '{name}' trong scripts/train_mamba_aqi.py")

    np.random.seed(seed)
    torch.manual_seed(seed)

    work_df = df.copy()
    if "location_key" not in work_df.columns:
        raise ValueError("Dataset train cần có cột location_key cho embedding.")

    work_df = work_df.loc[work_df["location_key"].astype(str).isin([str(x) for x in selected_locations])].copy()
    if work_df.empty:
        raise ValueError("Không có dữ liệu train cho các location đã chọn.")

    x_seq, loc_ids, y, y_ts, num_locations, numeric_cols = mod.build_time_series_samples(
        df=work_df,
        target_col=target_col,
        window_size=window_size,
        horizon=horizon,
        input_feature_cols=input_feature_cols,
        include_target_history=include_target_history,
    )
    target_feature_idx = numeric_cols.index(target_col) if target_col in numeric_cols else None
    train, val, test = mod.split_data_by_timeline(x_seq, loc_ids, y, y_ts)

    # Capture feature normalization stats BEFORE standardization for future-window inference.
    mean = train.x_seq.mean(axis=(0, 1), keepdims=True)
    std = train.x_seq.std(axis=(0, 1), keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)

    train, val, test, y_mean, y_std = mod.standardize(train, val, test)

    train_ds = mod.AQIDataset(train)
    val_ds = mod.AQIDataset(val)
    test_ds = mod.AQIDataset(test)
    pin_memory = use_gpu and torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")
    model = mod.TimeSeriesMambaRegressorNoLoc(
        num_features=train.x_seq.shape[-1],
        d_model=d_model,
        n_layers=n_layers,
        target_feature_idx=target_feature_idx,
    ).to(device)

    criterion = nn.HuberLoss(delta=1.0) if loss_name == "huber" else nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = max(1, epochs * max(1, len(train_loader)))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy="cos",
        div_factor=10.0,
        final_div_factor=100.0,
    )
    amp_enabled = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    best_val_loss = float("inf")
    best_state = None
    history = []

    total_steps = epochs * len(train_loader)
    global_step = 0
    prog = st.progress(0)
    log_box = st.empty()
    log_lines = []

    start_all = time.time()
    progress_update_interval = 20
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        epoch_start = time.time()
        optimizer.zero_grad(set_to_none=True)

        for step, (xb, loc_ids, yb) in enumerate(train_loader, start=1):
            xb = xb.to(device)
            loc_ids = loc_ids.to(device)
            yb = yb.to(device)

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                out = model(xb, loc_ids)
                loss = criterion(out, yb)

            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                continue

            if amp_enabled:
                scaler.scale(loss / grad_accum_steps).backward()
            else:
                (loss / grad_accum_steps).backward()

            if step % grad_accum_steps == 0 or step == len(train_loader):
                if max_grad_norm > 0:
                    if amp_enabled:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                if amp_enabled:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item() * yb.size(0)

            global_step += 1
            if total_steps > 0 and (step % progress_update_interval == 0 or step == len(train_loader)):
                prog.progress(min(global_step / total_steps, 1.0))

            if log_interval > 0 and (step % log_interval == 0 or step == len(train_loader)):
                avg_loss = running_loss / max(step * yb.size(0), 1)
                line = (
                    f"Epoch {epoch}/{epochs} | step {step}/{len(train_loader)} | "
                    f"batch_loss={loss.item():.6f} | running_avg={avg_loss:.6f}"
                )
                log_lines.append(line)
                if log_interval > 0:
                    log_box.code("\n".join(log_lines[-20:]))

        train_loss = running_loss / len(train_loader.dataset)
        val_metrics = evaluate(model, val_loader, criterion, device, y_mean, y_std)

        epoch_line = (
            f"Epoch {epoch}/{epochs} done | train_loss={train_loss:.6f} | val_loss={val_metrics['loss']:.6f} | "
            f"val_mae={val_metrics['mae']:.4f} | val_rmse={val_metrics['rmse']:.4f} | val_r2={val_metrics['r2']:.4f} | "
            f"sec={time.time() - epoch_start:.1f}"
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
                "train_sec": time.time() - epoch_start,
            }
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Không có checkpoint hợp lệ trong quá trình train.")

    model.load_state_dict(best_state)
    model.to(device)
    val_metrics = evaluate(model, val_loader, criterion, device, y_mean, y_std)
    test_metrics = evaluate(model, test_loader, criterion, device, y_mean, y_std)

    # Forecast base: user-provided test.csv preferred; fallback to internal split test.
    # Base for forecast is the latest timeline segment in filtered data.
    base_df = forecast_base_df if forecast_base_df is not None else work_df.copy()
    if not isinstance(base_df, pd.DataFrame) or base_df.empty:
        raise ValueError("Không có dữ liệu test làm mốc để dự báo 24h tiếp theo.")

    # 24h forecast after the last timestamp of each location in test base.
    future_df = build_future_24h_frame(base_df, feature_cols=numeric_cols, target_col=target_col)
    combined_df = pd.concat([base_df, future_df], ignore_index=True)
    combined_df["ts_utc"] = pd.to_datetime(combined_df["ts_utc"], utc=True, errors="coerce")
    combined_df = combined_df.sort_values("ts_utc").reset_index(drop=True)

    future_preds = _predict_future_from_rows(
        model=model,
        combined_df=combined_df,
        future_df=future_df,
        numeric_cols=numeric_cols,
        window_size=window_size,
        horizon=horizon,
        device=device,
        y_mean=y_mean,
        y_std=y_std,
        x_mean=mean,
        x_std=std,
    )

    # Keep forecast output minimal: generated hourly time + predicted target only.
    future_out = future_df[["ts_utc", "location_key"]].copy()
    future_out = future_out.rename(columns={"ts_utc": "time"})
    future_out[f"{target_col}_pred"] = future_preds
    future_out = future_out.sort_values(["location_key", "time"]).reset_index(drop=True)

    out_dir = run_dir
    if out_dir is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join("outputs", "streamlit_runs", run_id)
    os.makedirs(out_dir, exist_ok=True)

    model_path = os.path.join(out_dir, "best_mamba.pt")
    metrics_path = os.path.join(out_dir, "metrics_history.csv")
    future_pred_path = os.path.join(out_dir, forecast_file_name)

    torch.save(model.state_dict(), model_path)
    pd.DataFrame(history).to_csv(metrics_path, index=False)
    # Do NOT write the combined `future_24h_predictions.csv` file; produce per-location files only.
    # If an old combined file exists in the output dir (from previous runs), remove it.
    if os.path.exists(future_pred_path):
        try:
            os.remove(future_pred_path)
        except Exception:
            pass

    # Always emit per-location CSVs (time + prediction only) alongside the combined file.
    # This produces files like future_24h_predictions_<location>.csv without the `location_key` column.
    per_location_files = []
    if "location_key" in future_out.columns:
        for loc in sorted(future_out["location_key"].astype(str).unique().tolist()):
            loc_df = future_out.loc[future_out["location_key"].astype(str) == loc, ["time", f"{target_col}_pred"]].copy()
            loc_path = os.path.join(out_dir, f"future_24h_predictions_{sanitize_filename(loc)}.csv")
            loc_df.to_csv(loc_path, index=False)
            per_location_files.append(loc_path)

    summary = {
        "device": str(device),
        "n_rows_used": len(y),
        "split_train": len(train.y),
        "split_val": len(val.y),
        "split_test": len(test.y),
        "feature_count_after_encode": train.x_seq.shape[-1],
        "encoded_features": numeric_cols,
        "val_loss": val_metrics["loss"],
        "val_mae": val_metrics["mae"],
        "val_mse": float(val_metrics["rmse"] ** 2),
        "val_rmse": val_metrics["rmse"],
        "val_r2": val_metrics["r2"],
        "val_mape": _safe_mape(val_metrics["targets"], val_metrics["preds"]),
        "test_loss": test_metrics["loss"],
        "test_mae": test_metrics["mae"],
        "test_mse": float(test_metrics["rmse"] ** 2),
        "test_rmse": test_metrics["rmse"],
        "test_r2": test_metrics["r2"],
        "test_mape": _safe_mape(test_metrics["targets"], test_metrics["preds"]),
        "model_path": model_path,
        "metrics_path": metrics_path,
        "future_pred_path": future_pred_path,
        "future_rows": len(future_out),
        "future_locations": int(future_out["location_key"].nunique()),
        "per_location_files": per_location_files,
        "run_sec": time.time() - start_all,
        "sec_per_epoch": (time.time() - start_all) / max(epochs, 1),
    }
    return summary, pd.DataFrame(history), future_out


def train_transformer_pipeline(
    df: pd.DataFrame,
    selected_locations: list[str],
    target_col: str,
    window_size: int,
    horizon: int,
    epochs: int,
    batch_size: int,
    lr: float,
    d_model: int,
    seed: int,
    num_workers: int,
    use_gpu: bool,
    run_dir: str,
    input_feature_cols: list[str] | None = None,
    include_target_history: bool = True,
):
    mod = _load_train_module()
    if mod is None:
        raise RuntimeError("Không thể load scripts/train_mamba_aqi.py")
    TFT = _load_tft_class()
    if TFT is None:
        raise RuntimeError("Không thể load Transformer/models/temporal_fusion_t/tft_model.py")

    np.random.seed(seed)
    torch.manual_seed(seed)

    work_df = df.copy()
    work_df = work_df.loc[work_df["location_key"].astype(str).isin([str(x) for x in selected_locations])].copy()
    if work_df.empty:
        raise ValueError("Không có dữ liệu cho location đã chọn để train Transformer.")

    x_seq, loc_ids, y, y_ts, _, numeric_cols = mod.build_time_series_samples(
        df=work_df,
        target_col=target_col,
        window_size=window_size,
        horizon=horizon,
        input_feature_cols=input_feature_cols,
        include_target_history=include_target_history,
    )
    train, val, test = mod.split_data_by_timeline(x_seq, loc_ids, y, y_ts)

    x_mean = train.x_seq.mean(axis=(0, 1), keepdims=True)
    x_std = train.x_seq.std(axis=(0, 1), keepdims=True)
    x_std = np.where(x_std < 1e-6, 1.0, x_std)

    train, val, test, y_mean, y_std = mod.standardize(train, val, test)

    train_ds = mod.AQIDataset(train)
    val_ds = mod.AQIDataset(val)
    test_ds = mod.AQIDataset(test)
    pin_memory = use_gpu and torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")
    tft_params = {
        "total_time_steps": int(window_size + horizon),
        "input_size": int(train.x_seq.shape[-1]),
        "output_size": 1,
        "category_counts": "[]",
        "n_workers": int(num_workers),
        "input_obs_loc": "[0]",
        "static_input_loc": "[0]",
        "known_regular_inputs": str(list(range(1, int(train.x_seq.shape[-1])))).replace("'", "\""),
        "known_categorical_inputs": "[]",
        "quantiles": [0.5],
        "device": str(device),
        "hidden_layer_size": int(d_model),
        "dropout_rate": 0.1,
        "max_gradient_norm": 1.0,
        "lr": float(lr),
        "batch_size": int(batch_size),
        "num_epochs": int(epochs),
        "early_stopping_patience": 5,
        "num_encoder_steps": int(window_size),
        "stack_size": 1,
        "num_heads": 4,
    }
    if train.x_seq.shape[-1] <= 1:
        raise ValueError("TFT cần ít nhất 2 features số để tách known/observed inputs.")

    model = TFT(tft_params).to(device)
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_loss = float("inf")
    best_state = None
    history = []
    start_all = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        epoch_start = time.time()

        for xb, _, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            future_row = xb[:, -1:, :].clone()
            x_in = torch.cat([xb, future_row], dim=1)

            out, _, _ = model(x_in)
            pred = out[:, -1, 0]
            loss = criterion(pred, yb)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += loss.item() * yb.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        val_metrics = evaluate_tft(model, val_loader, criterion, device, y_mean, y_std)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "val_mae": val_metrics["mae"],
                "val_rmse": val_metrics["rmse"],
                "val_r2": val_metrics["r2"],
                "train_sec": time.time() - epoch_start,
            }
        )
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Không có checkpoint hợp lệ cho Transformer.")

    model.load_state_dict(best_state)
    model.to(device)
    val_metrics = evaluate_tft(model, val_loader, criterion, device, y_mean, y_std)
    test_metrics = evaluate_tft(model, test_loader, criterion, device, y_mean, y_std)

    base_df = work_df.copy()
    future_df = build_future_24h_frame(base_df, feature_cols=numeric_cols, target_col=target_col)
    combined_df = pd.concat([base_df, future_df], ignore_index=True)
    combined_df["ts_utc"] = pd.to_datetime(combined_df["ts_utc"], utc=True, errors="coerce")
    combined_df = combined_df.sort_values("ts_utc").reset_index(drop=True)

    future_preds = _predict_future_from_rows_tft(
        model=model,
        combined_df=combined_df,
        future_df=future_df,
        numeric_cols=numeric_cols,
        window_size=window_size,
        horizon=horizon,
        device=device,
        y_mean=y_mean,
        y_std=y_std,
        x_mean=x_mean,
        x_std=x_std,
    )

    future_out = future_df[["ts_utc", "location_key"]].copy().rename(columns={"ts_utc": "time"})
    future_out[f"{target_col}_pred"] = future_preds
    future_out = future_out.sort_values(["location_key", "time"]).reset_index(drop=True)

    os.makedirs(run_dir, exist_ok=True)
    model_path = os.path.join(run_dir, "best_transformer_tft.pt")
    metrics_path = os.path.join(run_dir, "metrics_history_transformer.csv")
    torch.save(model.state_dict(), model_path)
    pd.DataFrame(history).to_csv(metrics_path, index=False)

    per_location_files = []
    for loc in sorted(future_out["location_key"].astype(str).unique().tolist()):
        loc_df = future_out.loc[future_out["location_key"].astype(str) == loc, ["time", f"{target_col}_pred"]].copy()
        loc_path = os.path.join(run_dir, f"future_24h_predictions_transformer_{sanitize_filename(loc)}.csv")
        loc_df.to_csv(loc_path, index=False)
        per_location_files.append(loc_path)

    summary = {
        "device": str(device),
        "n_rows_used": len(y),
        "split_train": len(train.y),
        "split_val": len(val.y),
        "split_test": len(test.y),
        "feature_count_after_encode": train.x_seq.shape[-1],
        "encoded_features": numeric_cols,
        "val_loss": val_metrics["loss"],
        "val_mae": val_metrics["mae"],
        "val_mse": float(val_metrics["rmse"] ** 2),
        "val_rmse": val_metrics["rmse"],
        "val_r2": val_metrics["r2"],
        "val_mape": _safe_mape(val_metrics["targets"], val_metrics["preds"]),
        "test_loss": test_metrics["loss"],
        "test_mae": test_metrics["mae"],
        "test_mse": float(test_metrics["rmse"] ** 2),
        "test_rmse": test_metrics["rmse"],
        "test_r2": test_metrics["r2"],
        "test_mape": _safe_mape(test_metrics["targets"], test_metrics["preds"]),
        "model_path": model_path,
        "metrics_path": metrics_path,
        "future_pred_path": os.path.join(run_dir, "future_24h_predictions_transformer.csv"),
        "future_rows": len(future_out),
        "future_locations": int(future_out["location_key"].nunique()),
        "per_location_files": per_location_files,
        "run_sec": time.time() - start_all,
        "sec_per_epoch": (time.time() - start_all) / max(epochs, 1),
    }

    return summary, pd.DataFrame(history), future_out


def main():
    st.set_page_config(page_title="Mamba Trainer", layout="wide")
    st.title("Mamba Train/Val/Test Dashboard")
    st.caption("Chọn dataset, chọn target + input features, train/val/test 70/10/20, và xem log + dự đoán test.")

    with st.sidebar:
        st.header("Nguồn dữ liệu")
        source = st.radio("Dataset source", ["workspace path", "upload csv"], index=0)
        data_path = st.text_input("Path CSV trong workspace", value="dataset/2025.csv")
        uploaded = st.file_uploader("Hoặc upload CSV", type=["csv"])
        load_btn = st.button("Load dataset")

    if "df" not in st.session_state:
        st.session_state["df"] = None

    if load_btn:
        try:
            if source == "upload csv":
                if uploaded is None:
                    st.error("Bạn chưa upload file CSV.")
                else:
                    st.session_state["df"] = pd.read_csv(uploaded)
            else:
                st.session_state["df"] = pd.read_csv(data_path)
            st.success(f"Load thành công dataset: {st.session_state['df'].shape[0]} rows, {st.session_state['df'].shape[1]} cols")
        except Exception as e:
            st.error(f"Load dataset lỗi: {e}")

    df = st.session_state["df"]
    if df is None:
        st.info("Hãy bấm 'Load dataset' để bắt đầu.")
        return

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Preview dữ liệu")
        st.dataframe(df.head(20), use_container_width=True)
    with col2:
        st.subheader("Thông tin")
        st.write(f"Rows: {len(df):,}")
        st.write(f"Columns: {df.shape[1]}")

    all_cols = df.columns.tolist()
    if "location_key" not in df.columns:
        st.error("Dataset cần có cột location_key để tách train/dự báo theo từng địa điểm.")
        return

    locations = sorted(df["location_key"].dropna().astype(str).unique().tolist())
    if not locations:
        st.error("Không tìm thấy location_key hợp lệ trong dataset.")
        return

    reserved_cols = {"y_true", "y_pred", "abs_error"}
    target_options = [
        c for c in all_cols
        if c not in reserved_cols
        and c not in ["ts_utc", "location_key"]
        and not c.lower().startswith("unnamed:")
    ]

    # LOCATION selector first: choose locations, preview counts/splits, then show train config
    st.subheader("Chọn địa điểm để train + forecast riêng (trước khi cấu hình)")
    # Restrict to single location selection for per-location training
    selected_location = st.selectbox(
        "Chọn địa điểm để train + forecast riêng",
        options=locations,
        index=0,
        help="Chọn 1 địa điểm để train và forecast riêng cho tỉnh/thành đó.",
    )
    selected_locations = [selected_location]

    # quick window/horizon preview inputs used to estimate sample counts
    preview_col1, preview_col2 = st.columns([1, 1])
    with preview_col1:
        preview_window = st.number_input("Preview window size (timesteps)", min_value=1, max_value=168, value=24, step=1)
    with preview_col2:
        preview_horizon = st.number_input("Preview horizon", min_value=1, max_value=168, value=1, step=1)

    # show counts for selected locations using default target 'aqi' if available
    if selected_locations:
        try:
            df_sel = df.loc[df["location_key"].astype(str).isin([str(x) for x in selected_locations])].copy()
            # choose a reasonable default target for preview
            default_target = "aqi" if "aqi" in df_sel.columns else next((c for c in df_sel.select_dtypes(include=["number"]).columns if c not in ["_loc_id"]), None)
            if default_target is None:
                st.warning("Không tìm thấy cột số nào để preview sample counts. Cấu hình train sẽ yêu cầu chọn target.")
            else:
                mod = _load_train_module()
                if mod is None or not hasattr(mod, "build_time_series_samples"):
                    st.warning("Không thể load helper 'build_time_series_samples' để preview samples.")
                else:
                    x_seq, loc_ids, y, y_ts, num_locations, feature_cols = mod.build_time_series_samples(
                        df_sel, default_target, int(preview_window), int(preview_horizon)
                    )
                    if not hasattr(mod, "split_data_by_timeline"):
                        st.warning("Không thể load helper 'split_data_by_timeline' để preview samples.")
                    else:
                        train, val, test = mod.split_data_by_timeline(x_seq, loc_ids, y, y_ts)
                        st.markdown(f"**Preview (1 location)**: total samples={len(y):,}")
                        st.write(f"Train: {len(train.y):,}  |  Val: {len(val.y):,}  |  Test: {len(test.y):,}")
        except Exception as e:
            st.warning(f"Không thể tính preview samples: {e}")

    st.subheader("Cấu hình train")
    conf1, conf2, conf3 = st.columns(3)
    with conf1:
        target_col = st.selectbox(
            "Target column (biến cần dự đoán)",
            options=target_options,
            index=target_options.index("aqi") if "aqi" in target_options else 0,
        )
        metadata_cols = {"ts_utc", "location_key"}
        input_feature_options = [
            c for c in all_cols
            if c not in reserved_cols
            and c not in metadata_cols
            and not c.lower().startswith("unnamed:")
            and c != target_col
        ]
        selected_input_cols = st.multiselect(
            "Input features",
            options=input_feature_options,
            default=input_feature_options,
            help="Chọn các biến đầu vào (không gồm target).",
        )
        include_target_history = st.checkbox(
            "Dùng lịch sử target làm input lag",
            value=True,
            help="Bật để model học phần chênh lệch từ giá trị target gần nhất.",
        )
        loss_name = st.selectbox("Loss", options=["huber", "mse"], index=0)

    with conf2:
        window_size = st.number_input("Window size (T)", min_value=1, max_value=168, value=24, step=1)
        horizon = st.number_input("Horizon", min_value=1, max_value=168, value=1, step=1)
        epochs = st.number_input("Epochs", min_value=1, max_value=200, value=5, step=1)
        batch_size = st.number_input("Batch size", min_value=8, max_value=8192, value=128, step=8)
        lr = st.number_input("Learning rate", min_value=1e-6, max_value=1e-1, value=3e-4, format="%.6f")
        weight_decay = st.number_input("Weight decay", min_value=0.0, max_value=1.0, value=1e-4, format="%.6f")

    with conf3:
        d_model = st.number_input("d_model", min_value=16, max_value=512, value=64, step=16)
        n_layers = st.number_input("n_layers", min_value=1, max_value=8, value=2, step=1)
        grad_accum_steps = st.number_input("Gradient accumulation", min_value=1, max_value=64, value=2, step=1)
        max_grad_norm = st.number_input("Max grad norm", min_value=0.0, max_value=100.0, value=1.0, step=0.5)

    run1, run2, run3 = st.columns(3)
    with run1:
        seed = st.number_input("Seed", min_value=0, max_value=999999, value=42, step=1)
    with run2:
        num_workers = st.number_input("DataLoader workers", min_value=0, max_value=16, value=0, step=1)
    with run3:
        use_gpu = st.checkbox("Dùng GPU (nếu có)", value=True)

    st.info(
        "Tỉ lệ split cố định theo thời gian: Train 70% | Val 10% | Test 20%. "
        "Không dùng file test.csv riêng; test là các mốc thời gian gần nhất trong dataset tổng. "
        "Model train theo time-series B,T,F (single-location, no-embedding)."
    )

    if st.button("Train & Test", type="primary"):
        if len(selected_locations) == 0:
            st.error("Bạn cần chọn ít nhất 1 location.")
            return
        if len(selected_input_cols) == 0 and not include_target_history:
            st.error("Bạn cần chọn ít nhất 1 input feature hoặc bật target lag.")
            return

        with st.spinner("Đang train và evaluate..."):
            try:
                run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
                run_dir = os.path.join("outputs", "streamlit_runs", run_id)
                os.makedirs(run_dir, exist_ok=True)
                transformer_run_dir = os.path.join("outputs", "streamlit_runs", "Transformer_ouputs", run_id)
                os.makedirs(transformer_run_dir, exist_ok=True)

                # Use internal 20% latest-time split as test base (no external test.csv).
                forecast_base_df = None

                summary, hist_df, future_df = train_pipeline(
                    df=df,
                    forecast_base_df=forecast_base_df,
                    selected_locations=selected_locations,
                    target_col=target_col,
                    window_size=int(window_size),
                    horizon=int(horizon),
                    epochs=int(epochs),
                    batch_size=int(batch_size),
                    lr=float(lr),
                    weight_decay=float(weight_decay),
                    d_model=int(d_model),
                    n_layers=int(n_layers),
                    loss_name=loss_name,
                    seed=int(seed),
                    num_workers=int(num_workers),
                    use_gpu=bool(use_gpu),
                    log_interval=0,
                    grad_accum_steps=int(grad_accum_steps),
                    max_grad_norm=float(max_grad_norm),
                    input_feature_cols=selected_input_cols,
                    include_target_history=bool(include_target_history),
                    run_dir=run_dir,
                )

                t_summary, t_hist_df, t_future_df = train_transformer_pipeline(
                    df=df,
                    selected_locations=selected_locations,
                    target_col=target_col,
                    window_size=int(window_size),
                    horizon=int(horizon),
                    epochs=int(epochs),
                    batch_size=int(batch_size),
                    lr=float(lr),
                    d_model=int(d_model),
                    seed=int(seed),
                    num_workers=int(num_workers),
                    use_gpu=bool(use_gpu),
                    run_dir=transformer_run_dir,
                    input_feature_cols=selected_input_cols,
                    include_target_history=bool(include_target_history),
                )

                summary_df = pd.DataFrame([summary])

                train_counts = (
                    df.loc[df["location_key"].astype(str).isin([str(x) for x in selected_locations]), "location_key"]
                    .astype(str)
                    .value_counts()
                    .rename_axis("location_key")
                    .reset_index(name="train_source_rows")
                )
                test_counts = pd.DataFrame(
                    {
                        "location_key": selected_locations,
                        "test_source_rows": [int(summary["split_test"])],
                    }
                )

                used_counts = (
                    future_df["location_key"].astype(str).value_counts().rename_axis("location_key").reset_index(name="future_rows")
                )
                stats_df = train_counts.merge(test_counts, on="location_key", how="outer").merge(used_counts, on="location_key", how="outer")
                stats_df = stats_df.fillna(0)
            except Exception as e:
                st.error(f"Train/Test lỗi: {e}")
                return

        st.success("Train/Test hoàn tất")

        met1, met2, met3, met4 = st.columns(4)
        met1.metric("Val MAE", f"{summary['val_mae']:.4f}")
        met2.metric("Val RMSE", f"{summary['val_rmse']:.4f}")
        met3.metric("Val R2", f"{summary['val_r2']:.4f}")
        met4.metric("Locations done", f"{int(summary['future_locations']):,}")

        st.write("### So sánh Mamba vs Transformer (TFT)")
        compare_df = pd.DataFrame(
            [
                {
                    "model": "Mamba",
                    "train_total_sec": summary["run_sec"],
                    "sec_per_epoch": summary["sec_per_epoch"],
                    "val_mae": summary["val_mae"],
                    "val_mse": summary["val_mse"],
                    "val_rmse": summary["val_rmse"],
                    "val_r2": summary["val_r2"],
                    "val_mape": summary["val_mape"],
                    "test_mae": summary["test_mae"],
                    "test_mse": summary["test_mse"],
                    "test_rmse": summary["test_rmse"],
                    "test_r2": summary["test_r2"],
                    "test_mape": summary["test_mape"],
                },
                {
                    "model": "Transformer_TFT",
                    "train_total_sec": t_summary["run_sec"],
                    "sec_per_epoch": t_summary["sec_per_epoch"],
                    "val_mae": t_summary["val_mae"],
                    "val_mse": t_summary["val_mse"],
                    "val_rmse": t_summary["val_rmse"],
                    "val_r2": t_summary["val_r2"],
                    "val_mape": t_summary["val_mape"],
                    "test_mae": t_summary["test_mae"],
                    "test_mse": t_summary["test_mse"],
                    "test_rmse": t_summary["test_rmse"],
                    "test_r2": t_summary["test_r2"],
                    "test_mape": t_summary["test_mape"],
                },
            ]
        )
        st.dataframe(compare_df, use_container_width=True)

        # Quick winner cards for speed and accuracy
        faster_model = "Mamba" if summary["run_sec"] <= t_summary["run_sec"] else "Transformer_TFT"
        better_mae_model = "Mamba" if summary["test_mae"] <= t_summary["test_mae"] else "Transformer_TFT"
        better_mse_model = "Mamba" if summary["test_mse"] <= t_summary["test_mse"] else "Transformer_TFT"

        c1, c2, c3 = st.columns(3)
        c1.metric("Faster model", faster_model, f"Mamba: {summary['run_sec']:.2f}s | TFT: {t_summary['run_sec']:.2f}s")
        c2.metric("Best Test MAE", better_mae_model, f"Mamba: {summary['test_mae']:.4f} | TFT: {t_summary['test_mae']:.4f}")
        c3.metric("Best Test MSE", better_mse_model, f"Mamba: {summary['test_mse']:.4f} | TFT: {t_summary['test_mse']:.4f}")

        st.write("### Số dòng sau khi lọc theo địa điểm")
        merged_stats = stats_df.merge(
            pd.DataFrame(
                [
                    {
                        "n_rows_used": summary["n_rows_used"],
                        "split_train": summary["split_train"],
                        "split_val": summary["split_val"],
                        "split_test": summary["split_test"],
                    }
                ]
            ),
            how="cross",
        )
        st.dataframe(merged_stats, use_container_width=True)

        st.write("### Thống kê split")
        st.write(
            {
                "split_train": summary["split_train"],
                "split_val": summary["split_val"],
                "split_test": summary["split_test"],
                "n_rows_used": summary["n_rows_used"],
                "future_rows": summary["future_rows"],
                "future_locations": summary["future_locations"],
                "run_sec": round(summary["run_sec"], 2),
            }
        )

        st.write("### Lịch sử train")
        st.dataframe(hist_df, use_container_width=True)

        st.write("### Lịch sử train - Transformer (TFT)")
        st.dataframe(t_hist_df, use_container_width=True)

        st.write("### Dự báo 24 giờ tiếp theo (từng địa điểm)")
        st.dataframe(future_df.head(300), use_container_width=True)
        st.download_button(
            "Download file tổng (mọi location)",
            data=future_df.to_csv(index=False).encode("utf-8"),
            file_name="future_24h_predictions.csv",
            mime="text/csv",
        )

        st.info("Đã xuất file riêng cho location đã chọn trong thư mục run, ví dụ: future_24h_predictions_hcm.csv")
        if summary.get("per_location_files"):
            st.write("### Các file đã xuất")
            st.dataframe(pd.DataFrame({"file": summary["per_location_files"]}), use_container_width=True)

        st.write("### Dự báo 24 giờ tiếp theo - Transformer (TFT)")
        st.dataframe(t_future_df.head(300), use_container_width=True)
        if t_summary.get("per_location_files"):
            st.write("### Các file Transformer đã xuất")
            st.dataframe(pd.DataFrame({"file": t_summary["per_location_files"]}), use_container_width=True)

        st.code(
            "\n".join(
                [
                    f"run_dir: {os.path.dirname(summary['future_pred_path'])}",
                    "Files: future_24h_predictions_<location>.csv",
                    f"transformer_run_dir: {transformer_run_dir}",
                    "Files: future_24h_predictions_transformer_<location>.csv",
                ]
            )
        )


if __name__ == "__main__":
    main()

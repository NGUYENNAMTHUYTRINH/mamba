import os
import re
import sys
import time
import subprocess
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from mamba_ssm import Mamba
import importlib.util
from pathlib import Path
import yaml
import shutil


def normalize_locations(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [x.strip() for x in value.split(",") if x and x.strip()]
    if isinstance(value, (list, tuple, set)):
        out = []
        for item in value:
            s = str(item).strip()
            if s:
                out.append(s)
        return out
    s = str(value).strip()
    return [s] if s else []


def format_time_utc_strings(values: pd.Series) -> pd.Series:
    """Format datetimes as YYYY-MM-DD HH:MM:SS+00:00 in UTC."""
    ts = pd.to_datetime(values, utc=True, errors="coerce")
    return ts.dt.strftime("%Y-%m-%d %H:%M:%S+00:00")


def synthesize_tft_time_from_dataset(repo_root: Path, locations: pd.Series) -> pd.Series:
    """Create hourly UTC timestamps per location when TFT output has no usable time column."""
    dataset_path = repo_root / "dataset" / "2025.csv"
    loc_series = locations.astype(str).reset_index(drop=True)
    out = pd.Series(index=loc_series.index, dtype="object")

    base_map: dict[str, pd.Timestamp] = {}
    global_base = pd.Timestamp("2025-01-01 00:00:00", tz="UTC")

    if dataset_path.exists():
        try:
            src = pd.read_csv(dataset_path, usecols=["location_key", "ts_utc"])
            src["ts_utc"] = pd.to_datetime(src["ts_utc"], utc=True, errors="coerce")
            src = src.dropna(subset=["location_key", "ts_utc"]).copy()
            if not src.empty:
                max_per_loc = src.groupby(src["location_key"].astype(str))["ts_utc"].max()
                for k, v in max_per_loc.items():
                    base_map[str(k)] = v + pd.Timedelta(hours=1)
                global_base = src["ts_utc"].max() + pd.Timedelta(hours=1)
        except Exception:
            pass

    for loc, idx in loc_series.groupby(loc_series).groups.items():
        idx_list = list(idx)
        start = base_map.get(str(loc), global_base)
        rng = pd.date_range(start=start, periods=len(idx_list), freq="h", tz="UTC")
        out.loc[idx_list] = rng.strftime("%Y-%m-%d %H:%M:%S+00:00")

    return out


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


def run_tft_pipeline(
    selected_locations: list[str],
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    loss_name: str,
    seed: int,
    use_gpu: bool,
    run_dir: str,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Run Transformer_Timeseries TFT pipeline and return summary/history/predictions.

    Uses subprocess to avoid import path collisions between root repo and Transformer_Timeseries package layout.
    """
    repo_root = Path(__file__).parent
    tft_root = repo_root / "Transformer_Timeseries"
    if not tft_root.exists():
        tft_root = repo_root.parent / "Transformer_Timeseries"
    base_conf_path = tft_root / "conf" / "air_quality.yaml"
    if not base_conf_path.exists():
        raise FileNotFoundError(f"Không tìm thấy config TFT: {base_conf_path}")

    with open(base_conf_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["model"] = "tf_transformer"
    cfg["num_epochs"] = int(epochs)
    cfg["batch_size"] = int(batch_size)
    cfg["lr"] = float(lr)
    cfg["weight_decay"] = float(weight_decay)
    cfg["loss"] = str(loss_name)
    cfg["device"] = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"
    cfg["seed"] = int(seed)
    cfg["point_forecast"] = True
    cfg["use_quantile_loss_for_tft"] = False
    cfg["quantiles"] = [0.5]

    run_dir_abs = Path(run_dir).resolve()
    os.makedirs(run_dir_abs, exist_ok=True)
    runtime_conf_path = (run_dir_abs / "air_quality_tft_runtime.yaml").resolve()
    with open(runtime_conf_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    exp_name = f"streamlit_tft_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    selected_locations = normalize_locations(selected_locations)
    if not selected_locations:
        raise ValueError("TFT yêu cầu chọn ít nhất 1 location.")
    cmd = [
        sys.executable,
        "main.py",
        "--exp_name",
        exp_name,
        "--conf_file_path",
        str(runtime_conf_path),
        "--seed",
        str(int(seed)),
        "--locations",
        ",".join(selected_locations),
    ]

    started = time.time()
    tft_prog = st.progress(0)
    tft_log_box = st.empty()
    tft_log_lines = []
    epoch_pat = re.compile(r"Epoch\s+(\d+)\s*/\s*(\d+)", re.IGNORECASE)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    proc = subprocess.Popen(
        cmd,
        cwd=str(tft_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
        env=env,
    )

    stdout_lines = []
    if proc.stdout is not None:
        for raw_line in proc.stdout:
            line = raw_line.rstrip("\n")
            stdout_lines.append(line)
            if line.strip():
                tft_log_lines.append(line)
                tft_log_box.code("\n".join(tft_log_lines[-24:]))

            m = epoch_pat.search(line)
            if m:
                ep = int(m.group(1))
                ep_total = int(m.group(2))
                if ep_total > 0:
                    tft_prog.progress(min(ep / ep_total, 1.0))

    proc.wait()
    run_sec = time.time() - started
    proc_stdout = "\n".join(stdout_lines)
    proc_stderr = ""

    if proc.returncode == 0:
        tft_prog.progress(1.0)

    if proc.returncode != 0:
        raise RuntimeError(
            "TFT chạy lỗi.\n"
            f"Exit code: {proc.returncode}\n"
            f"STDOUT:\n{proc_stdout[-4000:]}\n\n"
            f"STDERR:\n{proc_stderr[-4000:]}"
        )

    exp_root = tft_root / "log" / cfg["model"] / exp_name
    if not exp_root.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục log TFT: {exp_root}")

    run_dirs = [p for p in exp_root.iterdir() if p.is_dir()]
    if not run_dirs:
        raise FileNotFoundError(f"Không có run folder trong: {exp_root}")
    latest_run_dir = max(run_dirs, key=lambda p: p.stat().st_mtime)

    pred_files = sorted(latest_run_dir.glob("transformer_predictions_*.csv"), key=lambda p: p.stat().st_mtime)
    if not pred_files:
        raise FileNotFoundError(f"Không tìm thấy file dự đoán TFT trong: {latest_run_dir}")

    pred_parts = []
    for p in pred_files:
        try:
            part = pd.read_csv(p)
            part["_source_file"] = p.name
            pred_parts.append(part)
        except Exception:
            continue
    if not pred_parts:
        raise ValueError("Không đọc được file dự đoán TFT hợp lệ.")
    pred_df = pd.concat(pred_parts, axis=0, ignore_index=True)

    if "location" in pred_df.columns:
        filtered_pred_df = pred_df[pred_df["location"].astype(str).isin(selected_locations)].copy()
        # Keep full dataframe when location labels are encoded ids instead of raw names.
        if not filtered_pred_df.empty:
            pred_df = filtered_pred_df

    pred_path = pred_files[-1]
    if not {"actual_aqi", "predicted_aqi"}.issubset(pred_df.columns):
        raise ValueError(f"File dự đoán TFT không đúng format: {pred_path}")

    if "location" not in pred_df.columns:
        pred_df["location"] = "unknown"
    if "time" not in pred_df.columns:
        pred_df["time"] = synthesize_tft_time_from_dataset(repo_root, pred_df["location"])
    else:
        parsed_time = format_time_utc_strings(pred_df["time"])
        if parsed_time.isna().all():
            pred_df["time"] = synthesize_tft_time_from_dataset(repo_root, pred_df["location"])
        else:
            fill_time = synthesize_tft_time_from_dataset(repo_root, pred_df["location"])
            pred_df["time"] = parsed_time.where(parsed_time.notna(), fill_time)

    future_pred_df = pd.DataFrame(
        {
            "time": format_time_utc_strings(pred_df["time"]),
            "location": pred_df["location"].astype(str),
            "predicted": pd.to_numeric(pred_df["predicted_aqi"], errors="coerce"),
        }
    ).dropna(subset=["predicted"])

    y_true = pd.to_numeric(pred_df["actual_aqi"], errors="coerce").dropna().to_numpy(dtype=np.float32)
    y_pred = pd.to_numeric(pred_df["predicted_aqi"], errors="coerce").dropna().to_numpy(dtype=np.float32)
    n = min(len(y_true), len(y_pred))
    if n == 0:
        raise ValueError("Không có dữ liệu dự đoán hợp lệ trong file TFT.")
    y_true = y_true[:n]
    y_pred = y_pred[:n]

    mse = mean_squared_error(y_true, y_pred)
    last_mae = float(mean_absolute_error(y_true, y_pred))
    last_rmse = float(np.sqrt(mse))
    last_r2 = float(r2_score(y_true, y_pred))
    summary = {
        "model": "tft",
        "device": cfg["device"],
        "seed": int(seed),
        "test_mae": last_mae,
        "test_rmse": last_rmse,
        "test_r2": last_r2,
        "run_sec": float(run_sec),
        "pred_path": str(pred_path),
        "log_dir": str(latest_run_dir),
        "selected_locations": ",".join(selected_locations),
        "stdout_tail": proc_stdout[-2000:],
        "tft_last_test_mae": last_mae,
        "tft_last_test_rmse": last_rmse,
        "tft_last_test_r2": last_r2,
        "train_only_sec": float(run_sec),
    }

    metrics_path = latest_run_dir / "metrics_history.csv"
    if metrics_path.exists():
        history_df = pd.read_csv(metrics_path)
        summary["metrics_path"] = str(metrics_path)
        if "train_sec" in history_df.columns:
            summary["train_only_sec"] = float(pd.to_numeric(history_df["train_sec"], errors="coerce").fillna(0).sum())

        required_cols = {"test_loss", "test_mae", "test_rmse", "test_r2"}
        if required_cols.issubset(set(history_df.columns)) and not history_df.empty:
            history_df = history_df.sort_values("epoch").reset_index(drop=True)
            best_idx = pd.to_numeric(history_df["test_loss"], errors="coerce").idxmin()
            best_row = history_df.loc[best_idx]
            last_row = history_df.iloc[-1]

            summary["tft_best_epoch"] = int(best_row.get("epoch", np.nan))
            summary["tft_best_test_loss"] = float(best_row.get("test_loss", np.nan))
            summary["tft_best_test_mae"] = float(best_row.get("test_mae", np.nan))
            summary["tft_best_test_rmse"] = float(best_row.get("test_rmse", np.nan))
            summary["tft_best_test_r2"] = float(best_row.get("test_r2", np.nan))

            summary["tft_hist_last_epoch"] = int(last_row.get("epoch", np.nan))
            summary["tft_hist_last_test_loss"] = float(last_row.get("test_loss", np.nan))
            summary["tft_hist_last_test_mae"] = float(last_row.get("test_mae", np.nan))
            summary["tft_hist_last_test_rmse"] = float(last_row.get("test_rmse", np.nan))
            summary["tft_hist_last_test_r2"] = float(last_row.get("test_r2", np.nan))

            # Default compare target for TFT is best-by-test-loss, mirroring checkpoint selection behavior.
            summary["test_mae"] = summary["tft_best_test_mae"]
            summary["test_rmse"] = summary["tft_best_test_rmse"]
            summary["test_r2"] = summary["tft_best_test_r2"]
            summary["selection"] = "best_test_loss"
        else:
            summary["selection"] = "last_prediction_file"
    else:
        history_df = pd.DataFrame(
            [{
                "epoch": cfg["num_epochs"],
                "test_mae": summary["test_mae"],
                "test_rmse": summary["test_rmse"],
                "test_r2": summary["test_r2"],
                "run_sec": summary["run_sec"],
            }]
        )
        summary["metrics_path"] = ""
        summary["selection"] = "last_prediction_file"

    # Standardized artifact export for TFT.
    out_dir = Path(run_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    best_ckpt_candidates = sorted(latest_run_dir.glob("*_best.pth"), key=lambda p: p.stat().st_mtime)
    if best_ckpt_candidates:
        best_src = best_ckpt_candidates[-1]
        best_dst = out_dir / "best_transformer_tft.pt"
        shutil.copy2(best_src, best_dst)
        summary["model_path"] = str(best_dst)
    else:
        summary["model_path"] = ""

    metrics_dst = out_dir / "metrics_history.csv"
    history_df.to_csv(metrics_dst, index=False)
    summary["metrics_path"] = str(metrics_dst)

    future_path = out_dir / "future_24h_predictions.csv"
    future_pred_df.to_csv(future_path, index=False)
    summary["future_pred_path"] = str(future_path)
    summary["future_rows"] = int(len(future_pred_df))
    summary["future_locations"] = int(future_pred_df["location"].nunique())
    summary["pred_path"] = str(future_path)

    return summary, history_df, pred_df


class SequenceDataset(Dataset):
    """Dataset for LSTM time series forecasting."""
    def __init__(self, X: np.ndarray, loc_ids: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.loc_ids = torch.from_numpy(loc_ids).long()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.loc_ids[idx], self.y[idx]


class LSTMForecaster(nn.Module):
    """LSTM model for time series forecasting."""
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, 
                 dropout: float = 0.2, horizon: int = 1, num_locations: int = 1, embed_dim: int = 8):
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
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_size, horizon)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def concat_location(self, x: torch.Tensor, loc_ids: torch.Tensor | None) -> torch.Tensor:
        if self.use_embedding:
            if loc_ids is None:
                raise ValueError("loc_ids is required when location embedding is enabled")
            loc_vec = self.location_emb(loc_ids)  # (B, E)
            loc_vec = loc_vec.unsqueeze(1).expand(-1, x.size(1), -1)  # (B, T, E)
            return torch.cat([x, loc_vec], dim=-1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, input_size + loc_embed_dim)
        output: (B, horizon)
        """
        out, _ = self.lstm(x)
        h_last = out[:, -1, :]  # (B, hidden_size)
        return self.fc(h_last)


class TabularDataset(Dataset):
    """Dataset for Mamba tabular forecasting."""
    def __init__(self, x: np.ndarray, loc_ids: np.ndarray, y: np.ndarray):
        self.x = torch.from_numpy(x).float()
        self.loc_ids = torch.from_numpy(loc_ids).long()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.loc_ids[idx], self.y[idx]


def make_lstm_windows(df: pd.DataFrame, feature_cols: list[str], target_col: str, 
                      lookback: int, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """Create sliding windows for LSTM from time series data."""
    features = df[feature_cols].values.astype(np.float32)
    target = df[target_col].values.astype(np.float32)
    
    X, y = [], []
    for i in range(len(features) - lookback - horizon + 1):
        X.append(features[i:i + lookback])
        y.append(target[i + lookback:i + lookback + horizon])
    
    if len(X) == 0:
        return np.array([]).reshape(0, lookback, len(feature_cols)), np.array([]).reshape(0, horizon)
    
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


@torch.no_grad()
def evaluate_lstm(model, loader, criterion, device, y_mean: float, y_std: float) -> dict:
    """Evaluate LSTM model."""
    model.eval()
    total_loss = 0.0
    preds = []
    targets = []

    loc_ids_all = []
    for xb, loc_ids, yb in loader:
        xb = xb.to(device)
        loc_ids = loc_ids.to(device)
        yb = yb.to(device)
        
        xb = model.concat_location(xb, loc_ids)
        out = model(xb)
        loss = criterion(out, yb)
        
        total_loss += loss.item() * yb.size(0)
        preds.append(out.detach().cpu().numpy())
        targets.append(yb.detach().cpu().numpy())
        loc_ids_all.append(loc_ids.detach().cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    targets = np.concatenate(targets, axis=0)
    
    # Inverse normalize
    preds_orig = preds * y_std + y_mean
    targets_orig = targets * y_std + y_mean
    
    mse = mean_squared_error(targets_orig, preds_orig)
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(targets_orig, preds_orig))
    r2 = float(r2_score(targets_orig.flatten(), preds_orig.flatten()))
    
    return {
        "loss": total_loss / len(loader.dataset),
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "preds": preds_orig,
        "targets": targets_orig,
        "loc_ids": np.concatenate(loc_ids_all, axis=0) if loc_ids_all else np.array([], dtype=np.int64),
    }


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
    """Run LSTM time series forecasting pipeline."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    selected_locations = normalize_locations(selected_locations)
    if not selected_locations:
        raise ValueError("LSTM yêu cầu chọn ít nhất 1 location")

    # Filter by selected locations
    df_loc = df.loc[df["location_key"].astype(str).isin(selected_locations)].copy()
    if df_loc.empty:
        raise ValueError(f"No data found for selected locations: {selected_locations}")
    
    df_loc["ts_utc"] = pd.to_datetime(df_loc["ts_utc"], utc=True, errors="coerce")
    df_loc = df_loc.dropna(subset=["ts_utc", "location_key"]).copy()
    df_loc = df_loc.sort_values(["location_key", "ts_utc"]).reset_index(drop=True)
    
    # Remove NaN values
    df_loc = df_loc.dropna(subset=feature_cols + [target_col])
    if len(df_loc) < lookback + horizon:
        raise ValueError(f"Not enough data for lookback={lookback} and horizon={horizon}")
    
    # Normalize features and target
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()
    
    df_scaled = df_loc.copy()
    df_scaled[feature_cols] = X_scaler.fit_transform(df_loc[feature_cols])
    y_scaled_all = y_scaler.fit_transform(df_loc[[target_col]])
    df_scaled[target_col] = y_scaled_all.flatten()
    
    # Create sequences per location (no cross-location mixing)
    loc_sorted = sorted(df_scaled["location_key"].astype(str).unique().tolist())
    loc_to_id = {loc: i for i, loc in enumerate(loc_sorted)}

    X_parts = []
    y_parts = []
    lid_parts = []
    loc_name_parts = []
    ts_parts = []
    for loc_name, g in df_scaled.groupby(df_scaled["location_key"].astype(str), sort=False):
        g = g.sort_values("ts_utc").reset_index(drop=True)
        X_loc, y_loc = make_lstm_windows(g, feature_cols, target_col, lookback, horizon)
        if len(X_loc) == 0:
            continue
        X_parts.append(X_loc)
        y_parts.append(y_loc)
        lid_parts.append(np.full(len(X_loc), loc_to_id[str(loc_name)], dtype=np.int64))
        loc_name_parts.append(np.full(len(X_loc), str(loc_name), dtype=object))
        ts_loc = g["ts_utc"].to_numpy(dtype="datetime64[ns]")
        ts_parts.append(ts_loc[lookback + horizon - 1:lookback + horizon - 1 + len(X_loc)])

    if not X_parts:
        raise ValueError("Could not create sequences from data")
    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    loc_ids = np.concatenate(lid_parts, axis=0)
    loc_names = np.concatenate(loc_name_parts, axis=0)
    sample_ts = np.concatenate(ts_parts, axis=0)
    
    # Train/Val/Test split by global timeline (70/10/20)
    order = np.argsort(sample_ts)
    X = X[order]
    y = y[order]
    loc_ids = loc_ids[order]
    loc_names = loc_names[order]

    n = len(X)
    train_end = int(0.7 * n)
    val_end = int(0.8 * n)
    
    X_train = X[:train_end]
    y_train = y[:train_end]
    X_val = X[train_end:val_end]
    y_val = y[train_end:val_end]
    X_test = X[val_end:]
    y_test = y[val_end:]
    lid_train = loc_ids[:train_end]
    lid_val = loc_ids[train_end:val_end]
    lid_test = loc_ids[val_end:]
    loc_name_test = loc_names[val_end:]
    test_time = sample_ts[val_end:]
    
    # Create dataloaders
    train_ds = SequenceDataset(X_train, lid_train, y_train)
    val_ds = SequenceDataset(X_val, lid_val, y_val)
    test_ds = SequenceDataset(X_test, lid_test, y_test)
    
    pin_memory = use_gpu and torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)
    
    device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")
    
    # Create model
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
    history = []
    
    prog = st.progress(0)
    log_box = st.empty()
    log_lines = []
    
    start_all = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        epoch_start = time.time()
        
        for step, (xb, lid, yb) in enumerate(train_loader, start=1):
            xb = xb.to(device)
            lid = lid.to(device)
            yb = yb.to(device)
            
            optimizer.zero_grad()
            xb = model.concat_location(xb, lid)
            out = model(xb)
            loss = criterion(out, yb)
            
            if torch.isfinite(loss):
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * yb.size(0)
        
        train_loss = running_loss / len(train_ds)
        val_metrics = evaluate_lstm(model, val_loader, criterion, device, y_scaler.mean_[0], y_scaler.scale_[0])
        
        epoch_sec = time.time() - epoch_start
        epoch_line = (
            f"Epoch {epoch}/{epochs} | train_loss={train_loss:.6f} | val_loss={val_metrics['loss']:.6f} | "
            f"val_mae={val_metrics['mae']:.4f} | val_rmse={val_metrics['rmse']:.4f} | val_r2={val_metrics['r2']:.4f} | "
            f"sec={epoch_sec:.1f}"
        )
        log_lines.append(epoch_line)
        log_box.code("\n".join(log_lines[-20:]))
        
        prog.progress(epoch / epochs)
        
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_mae": val_metrics["mae"],
            "val_rmse": val_metrics["rmse"],
            "val_r2": val_metrics["r2"],
            "train_sec": epoch_sec,
        })
        
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    
    if best_state is None:
        raise RuntimeError("No valid checkpoint found during training.")
    
    model.load_state_dict(best_state)
    model.to(device)
    
    val_metrics = evaluate_lstm(model, val_loader, criterion, device, y_scaler.mean_[0], y_scaler.scale_[0])
    test_metrics = evaluate_lstm(model, test_loader, criterion, device, y_scaler.mean_[0], y_scaler.scale_[0])
    
    # Create output directory
    if run_dir is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        lstm_run_dir = os.path.join("outputs", "lstm_runs", run_id)
    else:
        lstm_run_dir = run_dir
    os.makedirs(lstm_run_dir, exist_ok=True)
    
    # Save model and metrics
    model_path = os.path.join(lstm_run_dir, "best_lstm.pt")
    metrics_path = os.path.join(lstm_run_dir, "metrics_history.csv")
    pred_path = os.path.join(lstm_run_dir, "future_24h_predictions.csv")
    
    torch.save({
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
    }, model_path)
    
    pd.DataFrame(history).to_csv(metrics_path, index=False)
    
    # Create predictions DataFrame
    pred_df = pd.DataFrame({
        "time": format_time_utc_strings(pd.Series(test_time)),
        "location": loc_name_test,
        "predicted": test_metrics["preds"].flatten(),
    })
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



    def __init__(self, x: np.ndarray, loc_ids: np.ndarray, y: np.ndarray):
        self.x = torch.from_numpy(x).float()
        self.loc_ids = torch.from_numpy(loc_ids).long()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.loc_ids[idx], self.y[idx]


class TabularMambaRegressor(nn.Module):
    def __init__(self, num_features: int, num_locations: int, d_model: int = 64, n_layers: int = 2):
        super().__init__()
        self.scalar_proj = nn.Linear(1, d_model)
        self.location_emb = nn.Embedding(num_locations, d_model)
        self.layers = nn.ModuleList(
            [
                Mamba(
                    d_model=d_model,
                    d_state=16,
                    d_conv=4,
                    expand=2,
                    use_fast_path=False,
                )
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
        pooled = x.mean(dim=1)
        return self.head(pooled).squeeze(-1)


def unique_keep_order(items: list[str]) -> list[str]:
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def sanitize_filename(text: str) -> str:
    safe = []
    for ch in str(text).strip().lower():
        if ch.isalnum() or ch in ["_", "-"]:
            safe.append(ch)
        else:
            safe.append("_")
    name = "".join(safe).strip("_")
    return name or "unknown_location"


def encode_selected_features(df: pd.DataFrame, feature_cols: list[str]) -> tuple[np.ndarray, list[str]]:
    encoded_parts = []
    encoded_names = []

    for col in feature_cols:
        s = df[col]

        if pd.api.types.is_numeric_dtype(s):
            arr = pd.to_numeric(s, errors="coerce").astype("float32")
            fill_val = float(arr.median()) if not np.isnan(arr.median()) else 0.0
            arr = arr.fillna(fill_val).to_numpy().reshape(-1, 1)
            encoded_parts.append(arr)
            encoded_names.append(col)
            continue

        dt = pd.to_datetime(s, errors="coerce", utc=True)
        if dt.notna().mean() >= 0.6:
            hour = dt.dt.hour.fillna(0).astype("float32").to_numpy().reshape(-1, 1)
            dow = dt.dt.dayofweek.fillna(0).astype("float32").to_numpy().reshape(-1, 1)
            month = dt.dt.month.fillna(1).astype("float32").to_numpy().reshape(-1, 1)
            doy = dt.dt.dayofyear.fillna(1).astype("float32").to_numpy().reshape(-1, 1)
            encoded_parts.extend([hour, dow, month, doy])
            encoded_names.extend([f"{col}_hour", f"{col}_dayofweek", f"{col}_month", f"{col}_dayofyear"])
        else:
            cat = s.astype("category").cat.codes.replace(-1, 0).astype("float32").to_numpy().reshape(-1, 1)
            encoded_parts.append(cat)
            encoded_names.append(f"{col}_cat")

    if not encoded_parts:
        raise ValueError("Không có cột đầu vào hợp lệ sau khi encode.")

    x = np.concatenate(encoded_parts, axis=1).astype(np.float32)
    return x, encoded_names


def split_standardize(
    x: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
) -> dict:
    x_train = x[train_idx]
    x_val = x[val_idx]
    x_test = x[test_idx]

    y_train = y[train_idx]
    y_val = y[val_idx]
    y_test = y[test_idx]

    x_mean = x_train.mean(axis=0, keepdims=True)
    x_std = x_train.std(axis=0, keepdims=True)
    x_std = np.where(x_std < 1e-6, 1.0, x_std)

    y_mean = float(y_train.mean())
    y_std = float(y_train.std())
    if y_std < 1e-6:
        y_std = 1.0

    x_train = (x_train - x_mean) / x_std
    x_val = (x_val - x_mean) / x_std
    x_test = (x_test - x_mean) / x_std

    y_train = (y_train - y_mean) / y_std
    y_val = (y_val - y_mean) / y_std
    y_test = (y_test - y_mean) / y_std

    return {
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "x_mean": x_mean.astype(np.float32),
        "x_std": x_std.astype(np.float32),
        "x_train": x_train.astype(np.float32),
        "x_val": x_val.astype(np.float32),
        "x_test": x_test.astype(np.float32),
        "y_train": y_train.astype(np.float32),
        "y_val": y_val.astype(np.float32),
        "y_test": y_test.astype(np.float32),
        "y_mean": y_mean,
        "y_std": y_std,
    }


def split_data_by_timeline(
    x_seq: np.ndarray,
    loc_ids: np.ndarray,
    y: np.ndarray,
    y_ts: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
):
    """Fallback helper mirroring scripts/train_mamba_aqi.py split behavior."""
    if len(y) < 3:
        raise ValueError("Need at least 3 samples for train/val/test split.")

    order = np.argsort(y_ts)
    n = len(order)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    if train_end <= 0 or val_end <= train_end or val_end >= n:
        raise ValueError("Invalid timeline split sizes.")

    class _Split:
        def __init__(self, x, l, yy):
            self.x_seq = x
            self.loc_ids = l
            self.y = yy

    train_idx = order[:train_end]
    val_idx = order[train_end:val_end]
    test_idx = order[val_end:]
    return (
        _Split(x_seq[train_idx], loc_ids[train_idx], y[train_idx]),
        _Split(x_seq[val_idx], loc_ids[val_idx], y[val_idx]),
        _Split(x_seq[test_idx], loc_ids[test_idx], y[test_idx]),
    )


def make_split_indices(df_valid: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(df_valid)

    if "ts_utc" not in df_valid.columns:
        raise ValueError("Split theo thời gian cần cột 'ts_utc'.")

    ts = pd.to_datetime(df_valid["ts_utc"], utc=True, errors="coerce")
    if ts.isna().all():
        raise ValueError("Cột 'ts_utc' không parse được datetime.")

    if "location_key" in df_valid.columns:
        train_parts = []
        val_parts = []
        test_parts = []
        work = df_valid.copy()
        work["_ts"] = ts

        for _, group in work.sort_values("_ts").groupby("location_key", sort=False):
            g_idx = group.index.to_numpy()
            m = len(g_idx)
            if m < 5:
                # fallback cho nhóm quá nhỏ
                n_train = max(1, int(m * 0.7))
                n_val = max(1, int(m * 0.1))
                if n_train + n_val >= m:
                    n_val = 1
                    n_train = max(1, m - 2)
                n_test = m - n_train - n_val
                if n_test <= 0:
                    n_test = 1
                    n_train = max(1, n_train - 1)
            else:
                n_train = int(m * 0.7)
                n_val = int(m * 0.1)
                n_test = m - n_train - n_val

            train_parts.append(g_idx[:n_train])
            val_parts.append(g_idx[n_train:n_train + n_val])
            test_parts.append(g_idx[n_train + n_val:n_train + n_val + n_test])

        train_idx = np.concatenate(train_parts)
        val_idx = np.concatenate(val_parts)
        test_idx = np.concatenate(test_parts)
        return train_idx, val_idx, test_idx

    sorted_idx = np.argsort(ts.to_numpy())
    n_train = int(n * 0.7)
    n_val = int(n * 0.1)
    train_idx = sorted_idx[:n_train]
    val_idx = sorted_idx[n_train:n_train + n_val]
    test_idx = sorted_idx[n_train + n_val:]
    return train_idx, val_idx, test_idx


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

        xb = model.concat_location(xb, loc_ids)
        out = model(xb)
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
):
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Sequence Mamba pipeline: sliding windows per location, no cross-location sequence mixing.
    selected_locations = normalize_locations(selected_locations)
    if not selected_locations:
        raise ValueError("Cần chọn ít nhất 1 location để train Mamba.")

    work_df = df.copy()
    if "location_key" not in work_df.columns or "ts_utc" not in work_df.columns:
        raise ValueError("Dataset train Mamba cần có cột 'location_key' và 'ts_utc'.")

    work_df = work_df.loc[work_df["location_key"].astype(str).isin([str(x) for x in selected_locations])].copy()
    if work_df.empty:
        raise ValueError("Không có dữ liệu train cho các location đã chọn.")

    # Use canonical sequence helper from scripts/train_mamba_aqi.py when available.
    mod = _load_train_module()
    if mod is None or not hasattr(mod, "build_time_series_samples"):
        raise RuntimeError("Không load được module scripts/train_mamba_aqi.py để chạy Mamba sequence.")

    window_size = 24
    horizon = 1
    x_seq, loc_ids, y, y_ts, num_locations, ts_feature_cols = mod.build_time_series_samples(
        df=work_df,
        target_col=target_col,
        window_size=window_size,
        horizon=horizon,
    )

    train_split, val_split, test_split = mod.split_data_by_timeline(x_seq, loc_ids, y, y_ts)

    # Keep explicit scalers so future inference uses train-only statistics.
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

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    else:
        # Keep one core free for UI/process orchestration on CPU-only runs.
        cpu_threads = max(1, (os.cpu_count() or 2) - 1)
        torch.set_num_threads(cpu_threads)

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
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
    history = []
    amp_enabled = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    total_steps = epochs * len(train_loader)
    global_step = 0
    prog = st.progress(0)
    log_box = st.empty()
    log_lines = []

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
                xb = model.concat_location(xb, loc_batch)
                out = model(xb)
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

        epoch_line = (
            f"Epoch {epoch}/{epochs} done | train_loss={train_loss:.6f} | val_loss={val_metrics['loss']:.6f} | "
            f"val_mae={val_metrics['mae']:.4f} | val_rmse={val_metrics['rmse']:.4f} | val_r2={val_metrics['r2']:.4f} | "
            f"sec={time.time() - epoch_start:.1f}"
        )
        log_lines.append(epoch_line)
        log_box.code("\n".join(log_lines[-20:]))

        epoch_sec = time.time() - epoch_start
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

    # Build location -> id mapping consistent with category coding used in build_time_series_samples.
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

    # Forecast base: user-provided test.csv preferred; fallback to selected dataframe.
    base_df = forecast_base_df if forecast_base_df is not None else cleaned.copy()
    if not isinstance(base_df, pd.DataFrame) or base_df.empty:
        raise ValueError("Không có dữ liệu test làm mốc để dự báo 24h tiếp theo.")

    base_df = base_df.loc[base_df["location_key"].astype(str).isin([str(x) for x in loc_to_id.keys()])].copy()
    if base_df.empty:
        raise ValueError("Test CSV không có location trùng với dữ liệu train đã chọn.")

    # Fill numeric feature columns exactly like sequence sample builder.
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

    # Batched inference over prebuilt windows to reduce GPU kernel launch overhead.
    forecast_start = time.time()
    preds_rows = []
    model.eval()
    infer_x = []
    infer_loc = []
    infer_meta = []
    x_mean_2d = x_mean.squeeze(0)
    x_std_2d = x_std.squeeze(0)

    with torch.inference_mode():
        for loc in sorted(future_df["location_key"].astype(str).unique().tolist()):
            if loc not in loc_to_id:
                continue
            loc_hist = base_df.loc[base_df["location_key"].astype(str) == loc].copy()
            loc_hist["ts_utc"] = pd.to_datetime(loc_hist["ts_utc"], utc=True, errors="coerce")
            loc_hist = loc_hist.dropna(subset=["ts_utc"]).sort_values("ts_utc")

            if len(loc_hist) < window_size:
                continue

            rolling_window = loc_hist[ts_feature_cols].tail(window_size).to_numpy(dtype=np.float32)
            loc_future = future_df.loc[future_df["location_key"].astype(str) == loc].copy()
            loc_future["ts_utc"] = pd.to_datetime(loc_future["ts_utc"], utc=True, errors="coerce")
            loc_future = loc_future.sort_values("ts_utc")

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
                x_all = model.concat_location(x_all, loc_all)
                pred_norm_all = model(x_all).detach().float().cpu().numpy()

            pred_all = pred_norm_all * y_std + y_mean
            for (ts_val, loc_val), pred_val in zip(infer_meta, pred_all):
                preds_rows.append(
                    {
                        "time": ts_val,
                        "location": loc_val,
                        "predicted": float(pred_val),
                    }
                )

    forecast_sec = time.time() - forecast_start

    if not preds_rows:
        raise RuntimeError("Không tạo được dự báo 24h cho Mamba sequence.")

    future_out = pd.DataFrame(preds_rows)
    future_out["time"] = format_time_utc_strings(future_out["time"])
    future_out = future_out[["time", "location", "predicted"]].sort_values(["location", "time"]).reset_index(drop=True)

    out_dir = run_dir
    if out_dir is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join("outputs", "streamlit_runs", run_id)
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
        st.dataframe(df.head(20), width='stretch')
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
    feature_options = [
        c for c in all_cols
        if c not in reserved_cols
        and c not in ["ts_utc", "location_key"]
        and not c.lower().startswith("unnamed:")
    ]

    # LOCATION selector first: choose locations, preview counts/splits, then show train config
    st.subheader("Chọn địa điểm để train + forecast (trước khi cấu hình)")
    selected_locations = st.multiselect(
        "Chọn địa điểm để train + forecast",
        options=locations,
        default=locations[: min(3, len(locations))],
        help="Có thể chọn 1 hoặc nhiều địa điểm. Mô hình sẽ train chung theo phương pháp nhiều tỉnh.",
    )

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
                    # Use split function from loaded module when available
                    if hasattr(mod, "split_data_by_timeline"):
                        train, val, test = mod.split_data_by_timeline(x_seq, loc_ids, y, y_ts)
                    else:
                        train, val, test = split_data_by_timeline(x_seq, loc_ids, y, y_ts)
                    st.markdown(f"**Preview ({len(selected_locations)} locations)**: total samples={len(y):,}")
                    st.write(f"Train: {len(train.y):,}  |  Val: {len(val.y):,}  |  Test: {len(test.y):,}")
        except Exception as e:
            st.warning(f"Không thể tính preview samples: {e}")

    st.subheader("Cấu hình train")
    conf1, conf2, conf3 = st.columns(3)
    with conf1:
        target_col = st.selectbox(
            "Target column (biến cần dự đoán)",
            options=feature_options,
            index=feature_options.index("aqi") if "aqi" in feature_options else 0,
        )
        default_features = [c for c in feature_options if c != target_col]
        feature_cols = st.multiselect(
            "Input feature columns",
            options=[c for c in feature_options if c != target_col],
            default=default_features,
        )
        loss_name = st.selectbox("Loss", options=["huber", "mse"], index=0)

    with conf2:
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

    if use_gpu and not torch.cuda.is_available():
        st.warning(
            "Bạn đang bật GPU nhưng PyTorch hiện không nhận CUDA (torch+cpu). "
            "Train sẽ chạy bằng CPU nên thời gian mỗi epoch sẽ cao."
        )

    compare_with_tft = st.checkbox(
        "Chạy thêm TFT để so sánh",
        value=True,
        help="Khi bật, app sẽ train Mamba + TFT và hiển thị bảng compare test metrics trong cùng run.",
    )

    run1, run2, run3 = st.columns(3)
    with run1:
        compare_with_lstm = st.checkbox(
            "Chạy thêm LSTM để so sánh",
            value=True,
            help="Khi bật, app sẽ train LSTM với cùng tập locations đã chọn và hiển thị bảng compare.",
        )
    with run2:
        lstm_lookback = st.number_input("LSTM lookback", min_value=1, max_value=168, value=24, step=1)
    with run3:
        lstm_hidden = st.number_input("LSTM hidden size", min_value=16, max_value=512, value=64, step=16)

    lstm_col1, lstm_col2 = st.columns(2)
    with lstm_col1:
        lstm_num_layers = st.number_input("LSTM num_layers", min_value=1, max_value=8, value=2, step=1)
        lstm_dropout = st.number_input("LSTM dropout", min_value=0.0, max_value=0.9, value=0.2, format="%.2f")

    st.info(
        "Tỉ lệ split cố định theo thời gian: Train 70% | Val 10% | Test 20%. "
        "Không dùng file test.csv riêng; test là các mốc thời gian gần nhất trong dataset tổng."
    )


    if st.button("Train & Test", type="primary"):
        if len(feature_cols) == 0:
            st.error("Bạn cần chọn ít nhất 1 cột input.")
            return
        if len(selected_locations) == 0:
            st.error("Bạn cần chọn ít nhất 1 location.")
            return

        with st.spinner("Đang train và evaluate..."):
            try:
                run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
                mamba_run_dir = os.path.join("outputs", "mamba_runs", run_id)
                tft_run_dir = os.path.join("outputs", "transformers_runs", run_id)
                lstm_run_dir = os.path.join("outputs", "lstm_runs", run_id)
                os.makedirs(mamba_run_dir, exist_ok=True)
                os.makedirs(tft_run_dir, exist_ok=True)
                os.makedirs(lstm_run_dir, exist_ok=True)

                # Use internal 20% latest-time split as test base (no external test.csv).
                forecast_base_df = None

                summary, hist_df, future_df = train_pipeline(
                    df=df,
                    forecast_base_df=forecast_base_df,
                    selected_locations=selected_locations,
                    target_col=target_col,
                    feature_cols=feature_cols,
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
                    log_interval=50,
                    grad_accum_steps=int(grad_accum_steps),
                    max_grad_norm=float(max_grad_norm),
                    run_dir=mamba_run_dir,
                )

                tft_summary = None
                tft_hist_df = None
                tft_pred_df = None
                if compare_with_tft:
                    with st.spinner("Đang chạy TFT để so sánh..."):
                        tft_summary, tft_hist_df, tft_pred_df = run_tft_pipeline(
                            selected_locations=selected_locations,
                            epochs=int(epochs),
                            batch_size=int(batch_size),
                            lr=float(lr),
                            weight_decay=float(weight_decay),
                            loss_name=loss_name,
                            seed=int(seed),
                            use_gpu=bool(use_gpu),
                            run_dir=tft_run_dir,
                        )

                lstm_summary = None
                lstm_hist_df = None
                lstm_pred_df = None
                if compare_with_lstm:
                    with st.spinner("Đang chạy LSTM để so sánh..."):
                        lstm_summary, lstm_hist_df, lstm_pred_df = run_lstm_pipeline(
                            df=df,
                            selected_locations=selected_locations,
                            target_col=target_col,
                            feature_cols=feature_cols,
                            lookback=int(lstm_lookback),
                            horizon=1,
                            epochs=int(epochs),
                            batch_size=int(batch_size),
                            lr=float(lr),
                            hidden_size=int(lstm_hidden),
                            num_layers=int(lstm_num_layers),
                            dropout=float(lstm_dropout),
                            seed=int(seed),
                            use_gpu=bool(use_gpu),
                            run_dir=lstm_run_dir,
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
                        "test_source_rows": [int(summary["split_test"])] * len(selected_locations),
                    }
                )

                used_counts = (
                    future_df["location"].astype(str).value_counts().rename_axis("location_key").reset_index(name="future_rows")
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
        st.dataframe(merged_stats, width='stretch')

        st.write("### Thống kê split")
        st.write(
            {
                "split_train": summary["split_train"],
                "split_val": summary["split_val"],
                "split_test": summary["split_test"],
                "n_rows_used": summary["n_rows_used"],
                "future_rows": summary["future_rows"],
                "future_locations": summary["future_locations"],
                "train_only_sec": round(float(summary.get("train_only_sec", np.nan)), 2),
                "eval_sec": round(float(summary.get("eval_sec", np.nan)), 2),
                "forecast_sec": round(float(summary.get("forecast_sec", np.nan)), 2),
                "io_sec": round(float(summary.get("io_sec", np.nan)), 2),
                "run_sec": round(summary["run_sec"], 2),
            }
        )

        st.write("### Lịch sử train")
        st.dataframe(hist_df, width='stretch')

        if compare_with_tft and tft_summary is not None:
            st.write("### So sánh Mamba vs TFT (test metrics)")
            st.caption("Benchmark chuẩn: cùng loss, cùng seed; Mamba dùng best checkpoint theo val, TFT dùng best epoch theo test_loss (nếu có history).")
            compare_df = pd.DataFrame(
                [
                    {
                        "model": "mamba_best(val)",
                        "test_mae": summary.get("test_mae"),
                        "test_rmse": summary.get("test_rmse"),
                        "test_r2": summary.get("test_r2"),
                        "train_sec": summary.get("train_only_sec"),
                        "run_sec": summary.get("run_sec"),
                    },
                    {
                        "model": "tft_best(test_loss)",
                        "test_mae": tft_summary.get("test_mae"),
                        "test_rmse": tft_summary.get("test_rmse"),
                        "test_r2": tft_summary.get("test_r2"),
                        "train_sec": tft_summary.get("train_only_sec"),
                        "run_sec": tft_summary.get("run_sec"),
                    },
                ]
            )
            st.dataframe(compare_df, width='stretch')

            tft_modes_df = pd.DataFrame(
                [
                    {
                        "mode": "tft_best(test_loss)",
                        "epoch": tft_summary.get("tft_best_epoch"),
                        "test_mae": tft_summary.get("tft_best_test_mae"),
                        "test_rmse": tft_summary.get("tft_best_test_rmse"),
                        "test_r2": tft_summary.get("tft_best_test_r2"),
                    },
                    {
                        "mode": "tft_last(history)",
                        "epoch": tft_summary.get("tft_hist_last_epoch"),
                        "test_mae": tft_summary.get("tft_hist_last_test_mae", tft_summary.get("tft_last_test_mae")),
                        "test_rmse": tft_summary.get("tft_hist_last_test_rmse", tft_summary.get("tft_last_test_rmse")),
                        "test_r2": tft_summary.get("tft_hist_last_test_r2", tft_summary.get("tft_last_test_r2")),
                    },
                ]
            )
            st.write("### TFT benchmark modes")
            st.dataframe(tft_modes_df, width='stretch')

            if tft_hist_df is not None and not tft_hist_df.empty:
                st.write("### Lịch sử train TFT")
                st.dataframe(tft_hist_df, width='stretch')

            if tft_pred_df is not None and not tft_pred_df.empty:
                st.write("### TFT predictions preview")
                st.dataframe(tft_pred_df.head(100), width='stretch')

        if compare_with_lstm and lstm_summary is not None:
            st.write("### Lịch sử train LSTM")
            st.dataframe(lstm_hist_df, width='stretch')
            
            if lstm_pred_df is not None and not lstm_pred_df.empty:
                st.write("### LSTM predictions preview")
                st.dataframe(lstm_pred_df.head(100), width='stretch')

        # ========== 3-MODEL COMPARISON ==========
        if compare_with_tft and compare_with_lstm and tft_summary is not None and lstm_summary is not None:
            st.divider()
            st.write("### 📊 So sánh 3 mô hình (Mamba vs TFT vs LSTM)")
            st.caption("Đánh giá dựa trên test metrics - tất cả mô hình dùng cùng seed, cùng tập locations đã chọn")
            
            three_model_df = pd.DataFrame([
                {
                    "model": "mamba_best(val)",
                    "test_mae": summary.get("test_mae", np.nan),
                    "test_rmse": summary.get("test_rmse", np.nan),
                    "test_r2": summary.get("test_r2", np.nan),
                    "train_sec": summary.get("train_only_sec", np.nan),
                    "run_sec": summary.get("run_sec", np.nan),
                },
                {
                    "model": "tft_best(test_loss)",
                    "test_mae": tft_summary.get("test_mae", np.nan),
                    "test_rmse": tft_summary.get("test_rmse", np.nan),
                    "test_r2": tft_summary.get("test_r2", np.nan),
                    "train_sec": tft_summary.get("train_only_sec", np.nan),
                    "run_sec": tft_summary.get("run_sec", np.nan),
                },
                {
                    "model": "lstm_best(val)",
                    "test_mae": lstm_summary.get("test_mae", np.nan),
                    "test_rmse": lstm_summary.get("test_rmse", np.nan),
                    "test_r2": lstm_summary.get("test_r2", np.nan),
                    "train_sec": lstm_summary.get("train_only_sec", np.nan),
                    "run_sec": lstm_summary.get("run_sec", np.nan),
                },
            ])
            
            st.dataframe(three_model_df, width='stretch')
            
            # Find best model for each metric
            mae_best = three_model_df.loc[three_model_df["test_mae"].idxmin(), "model"]
            rmse_best = three_model_df.loc[three_model_df["test_rmse"].idxmin(), "model"]
            r2_best = three_model_df.loc[three_model_df["test_r2"].idxmax(), "model"]
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Best MAE", mae_best)
            col2.metric("Best RMSE", rmse_best)
            col3.metric("Best R²", r2_best)


        st.write("### Dự báo 24 giờ tiếp theo (từng địa điểm)")
        st.dataframe(future_df.head(300), width='stretch')
        st.download_button(
            "Download file tổng (mọi location)",
            data=future_df.to_csv(index=False).encode("utf-8"),
            file_name="future_24h_predictions.csv",
            mime="text/csv",
        )

        st.info("Mamba chỉ xuất 1 file tổng trong run_dir: future_24h_predictions.csv (gồm time, location, predicted).")

        st.code(
            "\n".join(
                [
                    f"run_dir: {os.path.dirname(summary['future_pred_path'])}",
                    "Files: future_24h_predictions.csv",
                ]
            )
        )


if __name__ == "__main__":
    main()
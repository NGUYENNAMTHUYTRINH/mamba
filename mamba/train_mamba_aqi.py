"""
mamba/train_mamba_aqi.py
------------------------
Script huấn luyện Mamba cho bài toán dự đoán AQI.

Import từ core:
    core.data_structs  → SplitData, AQIDataset
    core.metrics       → compute_metrics, denormalize
    core.utils         → setup_logger, resolve_device, set_seed

Import từ mamba:
    mamba.mamba_model  → TimeSeriesMambaRegressor

Chạy:
    python mamba/train_mamba_aqi.py --data-path dataset/2025.csv --epochs 10
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --- Đảm bảo Python tìm thấy thư mục gốc của project khi chạy trực tiếp ---
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data_structs import AQIDataset, SplitData
from core.metrics import compute_metrics, denormalize
from core.utils import resolve_device, set_seed, setup_logger
from mamba.mamba_model import TimeSeriesMambaRegressor


# ---------------------------------------------------------------------------
# Xây dựng samples time-series
# ---------------------------------------------------------------------------

def build_time_series_samples(
    df: pd.DataFrame,
    target_col: str,
    window_size: int,
    horizon: int,
    feature_cols: list[str] | None = None,
    include_target_history: bool = True,
):
    """Tạo sliding-window samples từ DataFrame time-series nhiều location.

    Parameters
    ----------
    df          : DataFrame gốc, cần có cột ts_utc, location_key, target_col
    target_col  : tên cột target cần dự đoán
    window_size : số timestep đầu vào (T)
    horizon     : dự đoán y(t + horizon)

    Returns
    -------
    x_seq        : (N, T, F) float32
    loc_ids      : (N,)      int64
    y            : (N,)      float32
    y_ts         : (N,)      datetime64[ns]
    num_locations: int
    feature_cols : list[str] — tên các cột feature được dùng
    """
    # Validate
    for col, label in [(target_col, "target"), ("ts_utc", "timestamp"), ("location_key", "location")]:
        if col not in df.columns:
            raise ValueError(f"Cột {label} '{col}' không tìm thấy trong dataset.")
    if window_size < 1:
        raise ValueError("window_size phải >= 1.")
    if horizon < 1:
        raise ValueError("horizon phải >= 1.")

    work = df.copy()
    work["_ts"] = pd.to_datetime(work["ts_utc"], utc=True, errors="coerce")
    missing_required = work[["_ts", "location_key", target_col]].isna().any(axis=1)
    if missing_required.any():
        raise ValueError("Dữ liệu chứa NaN ở ts_utc/location_key/target. Vui lòng làm sạch trước.")

    work["_loc_id"] = work["location_key"].astype("category").cat.codes.astype(np.int64)
    num_locations   = int(work["_loc_id"].max()) + 1

    # Chọn feature: ưu tiên feature_cols nếu được truyền vào
    if feature_cols is None:
        numeric_cols = work.select_dtypes(include=[np.number]).columns.tolist()
        for col in [target_col, "_loc_id"]:
            if col in numeric_cols:
                numeric_cols.remove(col)
        if include_target_history and target_col in work.columns:
            numeric_cols.append(target_col)
    else:
        numeric_cols = [c for c in feature_cols if c in work.columns and c != "_loc_id"]
        if include_target_history and target_col in work.columns and target_col not in numeric_cols:
            numeric_cols.append(target_col)
        if not include_target_history and target_col in numeric_cols:
            numeric_cols.remove(target_col)

    if not numeric_cols:
        raise ValueError("Không tìm thấy cột feature numeric nào sau khi lọc.")

    # Ép numeric và kiểm tra NaN
    for col in numeric_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    if work[numeric_cols].isna().any(axis=1).any():
        raise ValueError("Dữ liệu chứa NaN ở feature_cols. Vui lòng làm sạch trước.")

    work = work.sort_values(["_loc_id", "_ts"]).reset_index(drop=True)

    x_seq_list, loc_id_list, y_list, y_ts_list = [], [], [], []

    for loc_id, group in work.groupby("_loc_id", sort=False):
        x_vals  = group[numeric_cols].to_numpy(dtype=np.float32)
        y_vals  = group[target_col].to_numpy(dtype=np.float32)
        ts_vals = group["_ts"].to_numpy(dtype="datetime64[ns]")
        n = len(group)

        max_start = n - window_size - horizon + 1
        if max_start <= 0:
            continue

        for start in range(max_start):
            end        = start + window_size
            target_idx = end + horizon - 1
            x_seq_list.append(x_vals[start:end])
            loc_id_list.append(loc_id)
            y_list.append(y_vals[target_idx])
            y_ts_list.append(ts_vals[target_idx])

    if not x_seq_list:
        raise ValueError(
            "Không tạo được sample nào. "
            "Thử giảm --window-size / --horizon hoặc cung cấp nhiều dữ liệu hơn."
        )

    return (
        np.stack(x_seq_list).astype(np.float32),
        np.asarray(loc_id_list, dtype=np.int64),
        np.asarray(y_list, dtype=np.float32),
        np.asarray(y_ts_list, dtype="datetime64[ns]"),
        num_locations,
        numeric_cols,
    )


# ---------------------------------------------------------------------------
# Split theo timeline
# ---------------------------------------------------------------------------

def split_data_by_timeline(
    x_seq:   np.ndarray,
    loc_ids: np.ndarray,
    y:       np.ndarray,
    y_ts:    np.ndarray,
    train_ratio: float = 0.7,
    val_ratio:   float = 0.1,
) -> tuple[SplitData, SplitData, SplitData]:
    """Chia dữ liệu theo thứ tự thời gian (không shuffle).

    Train 70% → Val 10% → Test 20% (mặc định).
    """
    if len(y) < 3:
        raise ValueError("Cần ít nhất 3 sample để chia train/val/test.")

    order     = np.argsort(y_ts)
    n         = len(order)
    train_end = int(n * train_ratio)
    val_end   = train_end + int(n * val_ratio)

    if train_end <= 0 or val_end <= train_end or val_end >= n:
        raise ValueError(
            "Kích thước split không hợp lệ. Cần nhiều sample hơn hoặc điều chỉnh ratio."
        )

    def _take(idx):
        return SplitData(
            x_seq=x_seq[idx],
            loc_ids=loc_ids[idx],
            y=y[idx],
        )

    return (
        _take(order[:train_end]),
        _take(order[train_end:val_end]),
        _take(order[val_end:]),
    )


# ---------------------------------------------------------------------------
# Normalize
# ---------------------------------------------------------------------------

def standardize(
    train: SplitData,
    val:   SplitData,
    test:  SplitData,
) -> tuple[SplitData, SplitData, SplitData, float, float]:
    """Chuẩn hoá x_seq và y dựa trên thống kê của tập train.

    Returns
    -------
    train, val, test (đã normalize), y_mean, y_std
    """
    # X: normalize per-feature theo toàn bộ timestep của train
    x_mean = train.x_seq.mean(axis=(0, 1), keepdims=True)
    x_std  = train.x_seq.std(axis=(0, 1), keepdims=True)
    x_std  = np.where(x_std < 1e-6, 1.0, x_std)

    for split in [train, val, test]:
        split.x_seq = (split.x_seq - x_mean) / x_std

    # Y: normalize scalar
    y_mean = float(train.y.mean())
    y_std  = float(train.y.std())
    if y_std < 1e-6:
        y_std = 1.0

    for split in [train, val, test]:
        split.y = (split.y - y_mean) / y_std

    return train, val, test, y_mean, y_std


# ---------------------------------------------------------------------------
# Training engine (Mamba-specific: tqdm, amp, grad_accum)
# ---------------------------------------------------------------------------

def run_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    logger,
    epoch_idx:       int,
    total_epochs:    int,
    log_interval:    int,
    use_amp:         bool,
    grad_accum_steps: int,
    max_grad_norm:   float,
) -> tuple[float, float]:
    """Chạy 1 epoch train, trả về (train_loss, elapsed_seconds)."""
    model.train()
    running_loss = 0.0
    start_t      = time.time()
    amp_enabled  = use_amp and device.type == "cuda"
    scaler       = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(loader, desc=f"Train {epoch_idx}/{total_epochs}", leave=False)

    for step, (x_seq, loc_ids, y) in enumerate(pbar, start=1):
        x_seq, loc_ids, y = x_seq.to(device), loc_ids.to(device), y.to(device)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            pred = model(x_seq, loc_ids)
            loss = criterion(pred, y)
            loss_for_backward = loss / grad_accum_steps

        if not torch.isfinite(loss):
            logger.warning("Non-finite loss tại epoch %d step %d, bỏ qua batch này.", epoch_idx, step)
            optimizer.zero_grad(set_to_none=True)
            continue

        if amp_enabled:
            scaler.scale(loss_for_backward).backward()
        else:
            loss_for_backward.backward()

        if step % grad_accum_steps == 0 or step == len(loader):
            if amp_enabled:
                scaler.unscale_(optimizer)
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        running_loss += loss.item() * y.size(0)
        avg_loss      = running_loss / (step * y.size(0))
        pbar.set_postfix(loss=f"{loss.item():.5f}", avg=f"{avg_loss:.5f}")

        if log_interval > 0 and (step % log_interval == 0 or step == len(loader)):
            logger.info(
                "Epoch %d/%d | step %d/%d | batch_loss=%.6f | running_avg=%.6f",
                epoch_idx, total_epochs, step, len(loader), loss.item(), avg_loss,
            )

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss, time.time() - start_t


@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device,
    use_amp:  bool,
    y_mean:   float,
    y_std:    float,
) -> dict[str, float]:
    """Evaluate model trên một DataLoader, trả về dict metrics."""
    model.eval()
    total_loss = 0.0
    preds, targets = [], []
    amp_enabled = use_amp and device.type == "cuda"

    for x_seq, loc_ids, y in loader:
        x_seq, loc_ids, y = x_seq.to(device), loc_ids.to(device), y.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            pred = model(x_seq, loc_ids)
            loss = criterion(pred, y)
        total_loss += loss.item() * y.size(0)
        preds.append(pred.cpu().numpy())
        targets.append(y.cpu().numpy())

    preds_arr   = denormalize(np.concatenate(preds),   y_mean, y_std)
    targets_arr = denormalize(np.concatenate(targets), y_mean, y_std)

    preds_norm_arr = np.concatenate(preds).astype(np.float32).flatten()
    targets_norm_arr = np.concatenate(targets).astype(np.float32).flatten()

    metrics = compute_metrics(targets_arr, preds_arr)
    try:
        mse_norm = mean_squared_error(targets_norm_arr, preds_norm_arr)
        metrics["mae_norm"] = float(mean_absolute_error(targets_norm_arr, preds_norm_arr))
        metrics["rmse_norm"] = float(np.sqrt(mse_norm))
    except Exception:
        metrics["mae_norm"] = float("nan")
        metrics["rmse_norm"] = float("nan")
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train Mamba AQI forecasting model")
    parser.add_argument("--data-path",        type=str,   default="dataset/2025.csv")
    parser.add_argument("--target-col",       type=str,   default="aqi")
    parser.add_argument("--window-size",      type=int,   default=24)
    parser.add_argument("--horizon",          type=int,   default=1)
    parser.add_argument("--epochs",           type=int,   default=2)
    parser.add_argument("--batch-size",       type=int,   default=512)
    parser.add_argument("--lr",               type=float, default=1e-3)
    parser.add_argument("--weight-decay",     type=float, default=1e-4)
    parser.add_argument("--d-model",          type=int,   default=64)
    parser.add_argument("--n-layers",         type=int,   default=2)
    parser.add_argument("--seed",             type=int,   default=42)
    parser.add_argument("--num-workers",      type=int,   default=0)
    parser.add_argument("--out-dir",          type=str,   default="outputs")
    parser.add_argument("--device",           type=str,   default="cuda", choices=["cuda", "cpu", "auto"])
    parser.add_argument("--location",         type=str,   default=None,   help="[Deprecated] Dùng --locations thay thế")
    parser.add_argument("--locations",        type=str,   default=None,   help="Comma-separated location_key list")
    parser.add_argument("--log-interval",     type=int,   default=50)
    parser.add_argument("--loss",             type=str,   default="huber", choices=["mse", "huber"])
    parser.add_argument("--amp",              action="store_true")
    parser.add_argument("--grad-accum-steps", type=int,   default=1)
    parser.add_argument("--max-grad-norm",    type=float, default=1.0)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    logger = setup_logger(args.out_dir, name="train_mamba_aqi")
    set_seed(args.seed)

    # Load data
    logger.info("Loading dataset: %s", args.data_path)
    df = pd.read_csv(args.data_path)
    logger.info("Total rows: %d", len(df))

    # Filter locations
    selected = []
    if args.locations:
        selected = [x.strip() for x in args.locations.split(",") if x.strip()]
    elif args.location:
        selected = [args.location.strip()]

    if selected:
        if "location_key" not in df.columns:
            raise ValueError("Dataset không có cột 'location_key'.")
        df = df[df["location_key"].astype(str).isin(selected)].copy()
        if df.empty:
            raise ValueError(f"Không tìm thấy dữ liệu cho locations: {selected}")
        logger.info("Đã lọc %d locations: %s | còn %d rows", len(selected), selected, len(df))

    # Build samples
    x_seq, loc_ids, y, y_ts, num_locations, feature_cols = build_time_series_samples(
        df, args.target_col, args.window_size, args.horizon
    )
    logger.info("Features (%d): %s", len(feature_cols), feature_cols)
    logger.info("Samples: %d | Locations: %d", len(y), num_locations)

    # Split + Normalize
    train, val, test = split_data_by_timeline(x_seq, loc_ids, y, y_ts)
    train, val, test, y_mean, y_std = standardize(train, val, test)
    logger.info(
        "Split — train: %d | val: %d | test: %d",
        len(train.y), len(val.y), len(test.y),
    )

    # DataLoaders
    device      = resolve_device(args.device)
    pin_memory  = device.type == "cuda"
    use_amp     = args.amp and device.type == "cuda"
    loader_kwargs = dict(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=pin_memory)

    train_loader = DataLoader(AQIDataset(train), shuffle=False, **loader_kwargs)
    val_loader   = DataLoader(AQIDataset(val),   shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(AQIDataset(test),  shuffle=False, **loader_kwargs)

    # Model
    model = TimeSeriesMambaRegressor(
        num_features=train.x_seq.shape[-1],
        num_locations=num_locations,
        d_model=args.d_model,
        n_layers=args.n_layers,
    ).to(device)

    criterion = nn.HuberLoss(delta=1.0) if args.loss == "huber" else nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    logger.info("Device: %s | AMP: %s | grad_accum: %d", device, use_amp, args.grad_accum_steps)
    if device.type == "cuda":
        logger.info("GPU: %s", torch.cuda.get_device_name(0))

    # Training loop
    best_val_loss  = float("inf")
    best_path      = os.path.join(args.out_dir, "best_mamba_aqi.pt")
    history_path   = os.path.join(args.out_dir, "metrics_history.csv")

    with open(history_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(
            ["epoch", "train_loss", "val_loss", "mae", "rmse", "val_r2", "train_sec"]
        )

    for epoch in range(1, args.epochs + 1):
        train_loss, train_sec = run_epoch(
            model, train_loader, criterion, optimizer, device, logger,
            epoch, args.epochs, args.log_interval,
            use_amp, args.grad_accum_steps, args.max_grad_norm,
        )
        val_metrics = evaluate(model, val_loader, criterion, device, use_amp, y_mean, y_std)

        logger.info(
            "Epoch %02d/%02d | train=%.6f | val_loss=%.6f | mae=%.4f | rmse=%.4f | r2=%.4f | %.1fs",
            epoch, args.epochs,
            train_loss, val_metrics["loss"],
            val_metrics.get("mae_norm", float("nan")),
            val_metrics.get("rmse_norm", float("nan")),
            val_metrics["r2"],
            train_sec,
        )

        with open(history_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                epoch,
                f"{train_loss:.8f}",
                f"{val_metrics['loss']:.8f}",
                f"{val_metrics.get('mae_norm', float('nan')):.8f}",
                f"{val_metrics.get('rmse_norm', float('nan')):.8f}",
                f"{val_metrics['r2']:.8f}",
                f"{train_sec:.2f}",
            ])

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(model.state_dict(), best_path)
            logger.info("→ Checkpoint mới: %s", best_path)

    # Test
    model.load_state_dict(torch.load(best_path, map_location=device))
    test_metrics = evaluate(model, test_loader, criterion, device, use_amp, y_mean, y_std)

    logger.info("TEST | loss=%.6f | mae=%.4f | rmse=%.4f | r2=%.4f",
                test_metrics["loss"], test_metrics["mae"],
                test_metrics["rmse"], test_metrics["r2"])
    logger.info("Best model: %s", best_path)
    logger.info("History   : %s", history_path)


if __name__ == "__main__":
    main()
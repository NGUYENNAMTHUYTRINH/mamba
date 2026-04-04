import argparse
import csv
import logging
import os
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from mamba_ssm import Mamba


@dataclass
class SplitData:
    x_seq: np.ndarray
    loc_ids: np.ndarray
    y: np.ndarray


class AQIDataset(Dataset):
    def __init__(self, split: SplitData):
        self.x_seq = torch.from_numpy(split.x_seq).float()
        self.loc_ids = torch.from_numpy(split.loc_ids).long()
        self.y = torch.from_numpy(split.y).float()

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return self.x_seq[idx], self.loc_ids[idx], self.y[idx]


class TimeSeriesMambaRegressor(nn.Module):
    def __init__(
        self,
        num_features: int,
        num_locations: int,
        d_model: int = 64,
        n_layers: int = 2,
        dropout: float = 0.1,
        target_feature_idx: int | None = None,
    ):
        super().__init__()
        self.feature_proj = nn.Linear(num_features, d_model)
        self.location_emb = nn.Embedding(num_locations, d_model)
        self.layer_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.layers = nn.ModuleList(
            [
                Mamba(
                    d_model=d_model,
                    d_state=16,
                    d_conv=4,
                    expand=2,
                    use_fast_path=True,
                )
                for _ in range(n_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        self.num_features = num_features
        self.target_feature_idx = target_feature_idx

    def forward(self, x_seq: torch.Tensor, loc_ids: torch.Tensor) -> torch.Tensor:
        # x_seq: (B, T, F)
        x = self.feature_proj(x_seq)  # (B, T, d_model)
        loc_token = self.location_emb(loc_ids).unsqueeze(1)  # (B, 1, d_model)
        x = torch.cat([loc_token, x], dim=1)  # (B, T+1, d_model)

        for norm, layer in zip(self.layer_norms, self.layers):
            residual = x
            x = layer(norm(x))
            x = residual + self.dropout(x)

        x = self.norm(x)
        last_token = x[:, -1, :]
        mean_token = x[:, 1:, :].mean(dim=1)
        pred = self.head(torch.cat([last_token, mean_token], dim=-1)).squeeze(-1)
        if self.target_feature_idx is not None:
            pred = pred + x_seq[:, -1, self.target_feature_idx]
        return pred


def setup_logger(out_dir: str):
    logger = logging.getLogger("train_mamba_aqi")
    logger.setLevel(logging.INFO)
    logger.handlers = []

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(os.path.join(out_dir, "train.log"), mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


class HybridRegLoss(nn.Module):
    def __init__(self, mae_weight: float = 0.8):
        super().__init__()
        self.mae_weight = float(mae_weight)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        l1 = F.l1_loss(pred, target)
        mse = F.mse_loss(pred, target)
        return self.mae_weight * l1 + (1.0 - self.mae_weight) * mse


def resolve_device(device_arg: str):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda but CUDA is not available. Please check your GPU/PyTorch CUDA setup.")
    return torch.device(device_arg)


def build_time_series_samples(
    df: pd.DataFrame,
    target_col: str,
    window_size: int,
    horizon: int,
    input_feature_cols: list[str] | None = None,
    include_target_history: bool = True,
):
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset columns: {df.columns.tolist()}")
    if "ts_utc" not in df.columns:
        raise ValueError("Expected 'ts_utc' column for chronological time-series training.")
    if "location_key" not in df.columns:
        raise ValueError("Expected 'location_key' column in dataset for location embedding.")
    if window_size < 1:
        raise ValueError("window_size must be >= 1")
    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    work = df.copy()
    work["_ts"] = pd.to_datetime(work["ts_utc"], utc=True, errors="coerce")
    work = work.dropna(subset=["_ts", "location_key", target_col]).copy()
    if work.empty:
        raise ValueError("No valid rows after dropping missing ts_utc/location_key/target.")

    work["_loc_id"] = work["location_key"].astype("category").cat.codes.astype(np.int64)
    num_locations = int(work["_loc_id"].max()) + 1

    if input_feature_cols is None:
        # Keep numeric features and include target history as an input lag signal.
        numeric_cols = work.select_dtypes(include=[np.number]).columns.tolist()
    else:
        numeric_cols = [c for c in input_feature_cols if c in work.columns]
        if not numeric_cols:
            raise ValueError("input_feature_cols is empty after filtering existing columns.")
        non_numeric = [c for c in numeric_cols if not pd.api.types.is_numeric_dtype(work[c])]
        if non_numeric:
            raise ValueError(f"All selected input features must be numeric. Non-numeric: {non_numeric}")

    if include_target_history:
        if target_col not in numeric_cols:
            numeric_cols.append(target_col)
    elif target_col in numeric_cols:
        numeric_cols.remove(target_col)

    if "_loc_id" in numeric_cols:
        numeric_cols.remove("_loc_id")

    if not numeric_cols:
        raise ValueError("No numeric feature columns found after excluding target column.")

    # Fill missing values with median so we can keep all rows.
    for col in numeric_cols:
        median_val = work[col].median()
        if pd.isna(median_val):
            median_val = 0.0
        work[col] = work[col].fillna(median_val)

    work = work.sort_values(["_loc_id", "_ts"]).reset_index(drop=True)

    x_seq_list = []
    loc_id_list = []
    y_list = []
    y_ts_list = []

    for loc_id, group in work.groupby("_loc_id", sort=False):
        x_vals = group[numeric_cols].to_numpy(dtype=np.float32)
        y_vals = group[target_col].to_numpy(dtype=np.float32)
        ts_vals = group["_ts"].to_numpy(dtype="datetime64[ns]")
        n = len(group)

        max_start = n - window_size - horizon + 1
        if max_start <= 0:
            continue

        for start in range(max_start):
            end = start + window_size
            target_idx = end + horizon - 1
            x_seq_list.append(x_vals[start:end])
            loc_id_list.append(loc_id)
            y_list.append(y_vals[target_idx])
            y_ts_list.append(ts_vals[target_idx])

    if not x_seq_list:
        raise ValueError(
            "No time-series samples created. Reduce --window-size/--horizon or provide more rows per location."
        )

    x_seq = np.stack(x_seq_list).astype(np.float32)
    loc_ids = np.asarray(loc_id_list, dtype=np.int64)
    y = np.asarray(y_list, dtype=np.float32)
    y_ts = np.asarray(y_ts_list, dtype="datetime64[ns]")

    return x_seq, loc_ids, y, y_ts, num_locations, numeric_cols


class TimeSeriesMambaRegressorNoLoc(nn.Module):
    """Variant of the model that does NOT use a learned location token.
    Use when training on a single location (filtering the dataframe first).
    The forward signature keeps `loc_ids` for compatibility with existing loaders.
    """
    def __init__(
        self,
        num_features: int,
        d_model: int = 64,
        n_layers: int = 2,
        dropout: float = 0.1,
        target_feature_idx: int | None = None,
    ):
        super().__init__()
        self.feature_proj = nn.Linear(num_features, d_model)
        self.layer_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.layers = nn.ModuleList(
            [
                Mamba(
                    d_model=d_model,
                    d_state=16,
                    d_conv=4,
                    expand=2,
                    use_fast_path=True,
                )
                for _ in range(n_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        self.target_feature_idx = target_feature_idx

    def forward(self, x_seq: torch.Tensor, loc_ids: torch.Tensor) -> torch.Tensor:
        # x_seq: (B, T, F)
        x = self.feature_proj(x_seq)  # (B, T, d_model)
        for norm, layer in zip(self.layer_norms, self.layers):
            residual = x
            x = layer(norm(x))
            x = residual + self.dropout(x)
        x = self.norm(x)
        last_token = x[:, -1, :]
        mean_token = x.mean(dim=1)
        pred = self.head(torch.cat([last_token, mean_token], dim=-1)).squeeze(-1)
        if self.target_feature_idx is not None:
            pred = pred + x_seq[:, -1, self.target_feature_idx]
        return pred


def split_data_by_timeline(
    x_seq: np.ndarray,
    loc_ids: np.ndarray,
    y: np.ndarray,
    y_ts: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
):
    if len(y) < 3:
        raise ValueError("Need at least 3 samples for train/val/test split.")

    order = np.argsort(y_ts)
    n = len(order)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    if train_end <= 0 or val_end <= train_end or val_end >= n:
        raise ValueError("Invalid timeline split sizes. Need more samples or adjust ratios.")

    train_idx = order[:train_end]
    val_idx = order[train_end:val_end]
    test_idx = order[val_end:]

    train = SplitData(x_seq=x_seq[train_idx], loc_ids=loc_ids[train_idx], y=y[train_idx])
    val = SplitData(x_seq=x_seq[val_idx], loc_ids=loc_ids[val_idx], y=y[val_idx])
    test = SplitData(x_seq=x_seq[test_idx], loc_ids=loc_ids[test_idx], y=y[test_idx])
    return train, val, test


def standardize(train: SplitData, val: SplitData, test: SplitData):
    mean = train.x_seq.mean(axis=(0, 1), keepdims=True)
    std = train.x_seq.std(axis=(0, 1), keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)

    train.x_seq = (train.x_seq - mean) / std
    val.x_seq = (val.x_seq - mean) / std
    test.x_seq = (test.x_seq - mean) / std

    y_mean = float(train.y.mean())
    y_std = float(train.y.std())
    if y_std < 1e-6:
        y_std = 1.0

    train.y = (train.y - y_mean) / y_std
    val.y = (val.y - y_mean) / y_std
    test.y = (test.y - y_mean) / y_std

    return train, val, test, y_mean, y_std


def run_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    logger,
    epoch_idx,
    total_epochs,
    log_interval,
    use_amp,
    grad_accum_steps,
    max_grad_norm,
):
    model.train()
    running_loss = 0.0
    start_t = time.time()
    optimizer.zero_grad(set_to_none=True)
    amp_enabled = use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    pbar = tqdm(loader, desc=f"Train {epoch_idx}/{total_epochs}", leave=False)
    for step, (x_seq, loc_ids, y) in enumerate(pbar, start=1):
        x_seq = x_seq.to(device)
        loc_ids = loc_ids.to(device)
        y = y.to(device)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            pred = model(x_seq, loc_ids)
            loss = criterion(pred, y)
            loss_for_backward = loss / grad_accum_steps

        if not torch.isfinite(loss):
            logger.warning("Non-finite loss encountered at epoch %d step %d, skipping batch", epoch_idx, step)
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
        avg_loss = running_loss / (step * y.size(0))
        pbar.set_postfix(loss=f"{loss.item():.5f}", avg=f"{avg_loss:.5f}")

        if log_interval > 0 and (step % log_interval == 0 or step == len(loader)):
            logger.info(
                "Epoch %d/%d | step %d/%d | batch_loss=%.6f | running_avg=%.6f",
                epoch_idx,
                total_epochs,
                step,
                len(loader),
                loss.item(),
                avg_loss,
            )
    epoch_loss = running_loss / len(loader.dataset)
    elapsed = time.time() - start_t
    return epoch_loss, elapsed


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp, y_mean, y_std):
    model.eval()
    total_loss = 0.0
    preds = []
    targets = []
    amp_enabled = use_amp and device.type == "cuda"

    for x_seq, loc_ids, y in loader:
        x_seq = x_seq.to(device)
        loc_ids = loc_ids.to(device)
        y = y.to(device)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            pred = model(x_seq, loc_ids)
            loss = criterion(pred, y)

        total_loss += loss.item() * y.size(0)
        preds.append(pred.cpu().numpy())
        targets.append(y.cpu().numpy())

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
    }


def main():
    parser = argparse.ArgumentParser(description="Train/val/test AQI prediction with Mamba (single script)")
    parser.add_argument("--data-path", type=str, default="dataset/2025.csv")
    parser.add_argument("--target-col", type=str, default="aqi")
    parser.add_argument("--window-size", type=int, default=24, help="Input timesteps T for each sample")
    parser.add_argument("--horizon", type=int, default=1, help="Predict y(t+horizon) from last step in window")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default="outputs")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu", "auto"])
    parser.add_argument("--location", type=str, default=None, help="If set, filter data to this location_key and train without location embedding")
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--loss", type=str, default="hybrid", choices=["mse", "huber", "mae", "hybrid"])
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision (recommended on CUDA)")
    parser.add_argument("--grad-accum-steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    logger = setup_logger(args.out_dir)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    logger.info("Loading dataset from: %s", args.data_path)
    df = pd.read_csv(args.data_path)
    logger.info("Total rows loaded: %d", len(df))

    # If user requested single-location mode, filter data early so we build samples only for that location
    if args.location is not None:
        loc_str = str(args.location)
        if "location_key" not in df.columns:
            raise ValueError("--location specified but dataset has no 'location_key' column")
        df = df[df["location_key"].astype(str) == loc_str].copy()
        if df.empty:
            raise ValueError(f"No rows found for location '{loc_str}' in dataset")
        logger.info("Filtered dataset to location=%s, rows now: %d", loc_str, len(df))

    x_seq, loc_ids, y, y_ts, num_locations, feature_cols = build_time_series_samples(
        df=df,
        target_col=args.target_col,
        window_size=args.window_size,
        horizon=args.horizon,
    )
    target_feature_idx = feature_cols.index(args.target_col) if args.target_col in feature_cols else None
    logger.info("Using numeric features (%d): %s", len(feature_cols), feature_cols)
    logger.info("Window size: %d | Horizon: %d", args.window_size, args.horizon)
    logger.info("Total time-series samples: %d", len(y))
    logger.info("Number of unique locations: %d", num_locations)

    train, val, test = split_data_by_timeline(x_seq, loc_ids, y, y_ts)
    train, val, test, y_mean, y_std = standardize(train, val, test)

    logger.info("Split sizes:")
    logger.info("  train: %d (%.2f%%)", len(train.y), len(train.y) / len(y) * 100)
    logger.info("  val  : %d (%.2f%%)", len(val.y), len(val.y) / len(y) * 100)
    logger.info("  test : %d (%.2f%%)", len(test.y), len(test.y) / len(y) * 100)

    pin_memory = args.device in ["cuda", "auto"]
    train_loader = DataLoader(
        AQIDataset(train),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        AQIDataset(val),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        AQIDataset(test),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    device = resolve_device(args.device)
    if args.grad_accum_steps < 1:
        raise ValueError("--grad-accum-steps must be >= 1")

    use_amp = args.amp and device.type == "cuda"
    logger.info("Training device: %s", device)
    if device.type == "cuda":
        logger.info("GPU: %s", torch.cuda.get_device_name(0))
    logger.info("AMP enabled: %s", use_amp)
    logger.info("Gradient accumulation steps: %d", args.grad_accum_steps)

    if args.location is not None:
        # single-location mode: use model variant without location embedding
        model = TimeSeriesMambaRegressorNoLoc(
            num_features=train.x_seq.shape[-1],
            d_model=args.d_model,
            n_layers=args.n_layers,
            target_feature_idx=target_feature_idx,
        ).to(device)
    else:
        model = TimeSeriesMambaRegressor(
            num_features=train.x_seq.shape[-1],
            num_locations=num_locations,
            d_model=args.d_model,
            n_layers=args.n_layers,
            target_feature_idx=target_feature_idx,
        ).to(device)

    if args.loss == "huber":
        criterion = nn.HuberLoss(delta=1.0)
    elif args.loss == "mse":
        criterion = nn.MSELoss()
    elif args.loss == "mae":
        criterion = nn.L1Loss()
    else:
        criterion = HybridRegLoss(mae_weight=0.8)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_score = float("inf")
    best_path = os.path.join(args.out_dir, "best_mamba_aqi.pt")
    best_metric_name = "mae" if args.loss in {"mae", "hybrid"} else "loss"
    history_path = os.path.join(args.out_dir, "metrics_history.csv")
    with open(history_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_mae", "val_rmse", "val_r2", "train_sec"])

    for epoch in range(1, args.epochs + 1):
        train_loss, train_sec = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            logger,
            epoch,
            args.epochs,
            args.log_interval,
            use_amp,
            args.grad_accum_steps,
            args.max_grad_norm,
        )
        val_metrics = evaluate(model, val_loader, criterion, device, use_amp, y_mean, y_std)

        logger.info(
            "Epoch %02d/%02d | train_loss=%.6f | val_loss=%.6f | val_mae=%.4f | val_rmse=%.4f | val_r2=%.4f | train_sec=%.1f",
            epoch,
            args.epochs,
            train_loss,
            val_metrics["loss"],
            val_metrics["mae"],
            val_metrics["rmse"],
            val_metrics["r2"],
            train_sec,
        )

        with open(history_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                f"{train_loss:.8f}",
                f"{val_metrics['loss']:.8f}",
                f"{val_metrics['mae']:.8f}",
                f"{val_metrics['rmse']:.8f}",
                f"{val_metrics['r2']:.8f}",
                f"{train_sec:.2f}",
            ])

        current_score = float(val_metrics[best_metric_name])
        if current_score < best_val_score:
            best_val_score = current_score
            torch.save(model.state_dict(), best_path)
            logger.info("New best checkpoint saved by val_%s: %s", best_metric_name, best_path)

    model.load_state_dict(torch.load(best_path, map_location=device))
    test_metrics = evaluate(model, test_loader, criterion, device, use_amp, y_mean, y_std)

    logger.info("Best model evaluation on TEST set:")
    logger.info("  test_loss: %.6f", test_metrics["loss"])
    logger.info("  test_mae : %.4f", test_metrics["mae"])
    logger.info("  test_rmse: %.4f", test_metrics["rmse"])
    logger.info("  test_r2  : %.4f", test_metrics["r2"])
    logger.info("Saved best model to: %s", best_path)
    logger.info("Saved training history to: %s", history_path)
    logger.info("Saved raw logs to: %s", os.path.join(args.out_dir, "train.log"))


if __name__ == "__main__":
    main()

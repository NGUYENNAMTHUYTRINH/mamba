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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from mamba_ssm import Mamba


@dataclass
class SplitData:
    x_num: np.ndarray
    loc_ids: np.ndarray
    y: np.ndarray


class AQIDataset(Dataset):
    def __init__(self, split: SplitData):
        self.x_num = torch.from_numpy(split.x_num).float()
        self.loc_ids = torch.from_numpy(split.loc_ids).long()
        self.y = torch.from_numpy(split.y).float()

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return self.x_num[idx], self.loc_ids[idx], self.y[idx]


class TabularMambaRegressor(nn.Module):
    def __init__(self, num_numeric_features: int, num_locations: int, d_model: int = 64, n_layers: int = 2):
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
        self.num_numeric_features = num_numeric_features

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


def resolve_device(device_arg: str):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda but CUDA is not available. Please check your GPU/PyTorch CUDA setup.")
    return torch.device(device_arg)


def build_features(df: pd.DataFrame, target_col: str):
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset columns: {df.columns.tolist()}")

    work = df.copy()

    if "ts_utc" in work.columns:
        ts = pd.to_datetime(work["ts_utc"], utc=True, errors="coerce")
        work["hour"] = ts.dt.hour.fillna(0).astype(np.float32)
        work["dayofweek"] = ts.dt.dayofweek.fillna(0).astype(np.float32)
        work["month"] = ts.dt.month.fillna(1).astype(np.float32)
        work["dayofyear"] = ts.dt.dayofyear.fillna(1).astype(np.float32)

    if "location_key" not in work.columns:
        raise ValueError("Expected 'location_key' column in dataset for location embedding.")

    location_codes = work["location_key"].astype("category").cat.codes.to_numpy(dtype=np.int64)
    num_locations = int(location_codes.max()) + 1

    # Use all numeric columns except target as model input.
    numeric_cols = work.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)

    if not numeric_cols:
        raise ValueError("No numeric feature columns found after excluding target column.")

    # Fill missing values with median so we can keep all rows.
    for col in numeric_cols:
        median_val = work[col].median()
        if pd.isna(median_val):
            median_val = 0.0
        work[col] = work[col].fillna(median_val)

    # Target missing values are removed because supervised training needs labels.
    work = work.loc[~work[target_col].isna()].copy()
    location_codes = location_codes[work.index.to_numpy()]

    x_num = work[numeric_cols].to_numpy(dtype=np.float32)
    y = work[target_col].to_numpy(dtype=np.float32)

    return x_num, location_codes, y, num_locations, numeric_cols


def split_data(x_num: np.ndarray, loc_ids: np.ndarray, y: np.ndarray, seed: int = 42):
    indices = np.arange(len(y))

    train_idx, temp_idx = train_test_split(indices, test_size=0.30, random_state=seed, shuffle=True)
    # From remaining 30%, split to val/test => val=10% overall, test=20% overall.
    val_idx, test_idx = train_test_split(temp_idx, test_size=(2.0 / 3.0), random_state=seed, shuffle=True)

    train = SplitData(x_num=x_num[train_idx], loc_ids=loc_ids[train_idx], y=y[train_idx])
    val = SplitData(x_num=x_num[val_idx], loc_ids=loc_ids[val_idx], y=y[val_idx])
    test = SplitData(x_num=x_num[test_idx], loc_ids=loc_ids[test_idx], y=y[test_idx])
    return train, val, test


def standardize(train: SplitData, val: SplitData, test: SplitData):
    mean = train.x_num.mean(axis=0, keepdims=True)
    std = train.x_num.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)

    train.x_num = (train.x_num - mean) / std
    val.x_num = (val.x_num - mean) / std
    test.x_num = (test.x_num - mean) / std

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
    for step, (x_num, loc_ids, y) in enumerate(pbar, start=1):
        x_num = x_num.to(device)
        loc_ids = loc_ids.to(device)
        y = y.to(device)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            pred = model(x_num, loc_ids)
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

    for x_num, loc_ids, y in loader:
        x_num = x_num.to(device)
        loc_ids = loc_ids.to(device)
        y = y.to(device)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            pred = model(x_num, loc_ids)
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
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--loss", type=str, default="huber", choices=["mse", "huber"])
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

    x_num, loc_ids, y, num_locations, feature_cols = build_features(df, args.target_col)
    logger.info("Using numeric features (%d): %s", len(feature_cols), feature_cols)
    logger.info("Number of unique locations: %d", num_locations)

    train, val, test = split_data(x_num, loc_ids, y, seed=args.seed)
    train, val, test, y_mean, y_std = standardize(train, val, test)

    logger.info("Split sizes:")
    logger.info("  train: %d (%.2f%%)", len(train.y), len(train.y) / len(y) * 100)
    logger.info("  val  : %d (%.2f%%)", len(val.y), len(val.y) / len(y) * 100)
    logger.info("  test : %d (%.2f%%)", len(test.y), len(test.y) / len(y) * 100)

    pin_memory = args.device in ["cuda", "auto"]
    train_loader = DataLoader(
        AQIDataset(train),
        batch_size=args.batch_size,
        shuffle=True,
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

    model = TabularMambaRegressor(
        num_numeric_features=train.x_num.shape[1],
        num_locations=num_locations,
        d_model=args.d_model,
        n_layers=args.n_layers,
    ).to(device)

    criterion = nn.HuberLoss(delta=1.0) if args.loss == "huber" else nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_loss = float("inf")
    best_path = os.path.join(args.out_dir, "best_mamba_aqi.pt")
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

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(model.state_dict(), best_path)
            logger.info("New best checkpoint saved: %s", best_path)

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

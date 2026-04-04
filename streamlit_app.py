import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from mamba_ssm import Mamba


class TabularDataset(Dataset):
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


def make_split_indices(df_valid: pd.DataFrame, seed: int, split_mode: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(df_valid)
    indices = np.arange(n)

    if split_mode == "random":
        train_idx, temp_idx = train_test_split(indices, test_size=0.30, random_state=seed, shuffle=True)
        val_idx, test_idx = train_test_split(temp_idx, test_size=(2.0 / 3.0), random_state=seed, shuffle=True)
        return train_idx, val_idx, test_idx

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
    split_mode: str,
    run_dir: str | None = None,
    forecast_file_name: str = "future_24h_predictions.csv",
    export_per_location_files: bool = False,
):
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Defensive sanitize to avoid duplicated labels downstream (e.g. location_key in multiple places).
    feature_cols = unique_keep_order([c for c in feature_cols if c in df.columns and c != target_col])
    if not feature_cols:
        raise ValueError("Không có feature hợp lệ sau khi loại target/column không tồn tại.")

    work_df = df.copy()
    if "location_key" not in work_df.columns:
        raise ValueError("Dataset train cần có cột location_key cho embedding.")

    work_df = work_df.loc[work_df["location_key"].astype(str).isin([str(x) for x in selected_locations])].copy()
    if work_df.empty:
        raise ValueError("Không có dữ liệu train cho các location đã chọn.")

    x_all, encoded_feature_names = encode_selected_features(work_df, feature_cols)
    y_all = pd.to_numeric(work_df[target_col], errors="coerce")

    valid_mask = ~y_all.isna()
    x_all = x_all[valid_mask.to_numpy()]
    y_all = y_all[valid_mask].to_numpy(dtype=np.float32)
    df_valid = work_df.loc[valid_mask].copy().reset_index(drop=True)

    locations_sorted = sorted(df_valid["location_key"].astype(str).unique().tolist())
    loc_to_id = {loc: i for i, loc in enumerate(locations_sorted)}
    loc_ids_all = df_valid["location_key"].astype(str).map(loc_to_id).to_numpy(dtype=np.int64)

    train_idx, val_idx, test_idx = make_split_indices(df_valid, seed=seed, split_mode=split_mode)

    split = split_standardize(x_all, y_all, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)

    loc_train = loc_ids_all[train_idx]
    loc_val = loc_ids_all[val_idx]

    train_ds = TabularDataset(split["x_train"], loc_train, split["y_train"])
    val_ds = TabularDataset(split["x_val"], loc_val, split["y_val"])
    pin_memory = use_gpu and torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")
    model = TabularMambaRegressor(
        num_features=split["x_train"].shape[1],
        num_locations=len(loc_to_id),
        d_model=d_model,
        n_layers=n_layers,
    ).to(device)

    criterion = nn.HuberLoss(delta=1.0) if loss_name == "huber" else nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = float("inf")
    best_state = None
    history = []

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

        for step, (xb, loc_ids, yb) in enumerate(train_loader, start=1):
            xb = xb.to(device)
            loc_ids = loc_ids.to(device)
            yb = yb.to(device)

            out = model(xb, loc_ids)
            loss = criterion(out, yb)

            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                continue

            (loss / grad_accum_steps).backward()

            if step % grad_accum_steps == 0 or step == len(train_loader):
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item() * yb.size(0)

            global_step += 1
            if total_steps > 0:
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
        val_metrics = evaluate(model, val_loader, criterion, device, split["y_mean"], split["y_std"])

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
    val_metrics = evaluate(model, val_loader, criterion, device, split["y_mean"], split["y_std"])

    # Forecast base: user-provided test.csv preferred; fallback to internal split test.
    base_df = forecast_base_df if forecast_base_df is not None else df_valid.iloc[split["test_idx"]].copy()
    if not isinstance(base_df, pd.DataFrame) or base_df.empty:
        raise ValueError("Không có dữ liệu test làm mốc để dự báo 24h tiếp theo.")

    base_df = base_df.loc[base_df["location_key"].astype(str).isin(locations_sorted)].copy()
    if base_df.empty:
        raise ValueError("Test CSV không có location trùng với dữ liệu train đã chọn.")

    # 24h forecast after the last timestamp of each location in test base.
    future_df = build_future_24h_frame(base_df, feature_cols=feature_cols, target_col=target_col)
    x_future, _ = encode_selected_features(future_df, feature_cols)
    x_future = ((x_future - split["x_mean"]) / split["x_std"]).astype(np.float32)

    loc_future = future_df["location_key"].astype(str).map(loc_to_id)
    if loc_future.isna().any():
        missing = sorted(future_df.loc[loc_future.isna(), "location_key"].astype(str).unique().tolist())
        raise ValueError(f"Có location trong test không tồn tại trong train: {missing[:5]}")

    future_ds = TabularDataset(
        x_future,
        loc_future.to_numpy(dtype=np.int64),
        np.zeros(len(x_future), dtype=np.float32),
    )
    future_loader = DataLoader(future_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin_memory)

    model.eval()
    future_preds = []
    with torch.no_grad():
        for xb, loc_ids, _ in future_loader:
            xb = xb.to(device)
            loc_ids = loc_ids.to(device)
            out = model(xb, loc_ids)
            future_preds.append(out.detach().cpu().numpy())

    future_preds = np.concatenate(future_preds, axis=0)
    future_preds = future_preds * split["y_std"] + split["y_mean"]

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
    future_out.to_csv(future_pred_path, index=False)

    per_location_files = []
    if export_per_location_files:
        for loc in locations_sorted:
            loc_df = future_out.loc[future_out["location_key"].astype(str) == loc, ["time", f"{target_col}_pred"]].copy()
            loc_path = os.path.join(out_dir, f"future_24h_predictions_{sanitize_filename(loc)}.csv")
            loc_df.to_csv(loc_path, index=False)
            per_location_files.append(loc_path)

    summary = {
        "device": str(device),
        "n_rows_used": len(y_all),
        "split_train": len(split["y_train"]),
        "split_val": len(split["y_val"]),
        "split_test": len(split["y_test"]),
        "feature_count_after_encode": split["x_train"].shape[1],
        "encoded_features": encoded_feature_names,
        "val_loss": val_metrics["loss"],
        "val_mae": val_metrics["mae"],
        "val_rmse": val_metrics["rmse"],
        "val_r2": val_metrics["r2"],
        "model_path": model_path,
        "metrics_path": metrics_path,
        "future_pred_path": future_pred_path,
        "future_rows": len(future_out),
        "future_locations": int(future_out["location_key"].nunique()),
        "per_location_files": per_location_files,
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
    feature_options = [
        c for c in all_cols
        if c not in reserved_cols
        and c not in ["ts_utc", "location_key"]
        and not c.lower().startswith("unnamed:")
    ]

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

    split_mode = st.selectbox(
        "Kiểu chia dữ liệu",
        options=["time", "random"],
        index=0,
        help="time: chia theo thời gian (khuyến nghị khi muốn dự báo tương lai 24h).",
    )

    selected_locations = st.multiselect(
        "Chọn địa điểm để train + forecast riêng",
        options=locations,
        default=locations[:1],
        help="Mặc định chọn 1 địa điểm. Bạn có thể chọn thêm nhiều địa điểm nếu muốn.",
    )

    forecast_test_path = st.text_input(
        "Test CSV để làm mốc forecast +24h",
        value="dataset/test.csv",
        help="Ví dụ test là ngày 20 thì model sẽ dự báo ngày 21 theo từng location.",
    )

    export_per_location_files = st.checkbox(
        "Xuất thêm file riêng từng location",
        value=False,
        help="Mặc định chỉ xuất 1 file tổng future_24h_predictions.csv có cột location_key.",
    )

    st.info("Tỉ lệ split cố định: Train 70% | Val 10% | Test 20%")

    if st.button("Train & Test", type="primary"):
        if len(feature_cols) == 0:
            st.error("Bạn cần chọn ít nhất 1 cột input.")
            return
        if len(selected_locations) == 0:
            st.error("Bạn cần chọn ít nhất 1 location.")
            return

        with st.spinner("Đang train và evaluate..."):
            try:
                forecast_base_df_all = None
                if forecast_test_path.strip():
                    if not os.path.exists(forecast_test_path):
                        st.error(f"Không tìm thấy file test: {forecast_test_path}")
                        return
                    forecast_base_df_all = pd.read_csv(forecast_test_path)

                run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
                run_dir = os.path.join("outputs", "streamlit_runs", run_id)
                os.makedirs(run_dir, exist_ok=True)

                if forecast_base_df_all is not None and "location_key" not in forecast_base_df_all.columns:
                    st.error("Test CSV cần có cột location_key để lọc theo địa điểm.")
                    return

                forecast_base_df = forecast_base_df_all
                if forecast_base_df is not None:
                    forecast_base_df = forecast_base_df.loc[
                        forecast_base_df["location_key"].astype(str).isin([str(x) for x in selected_locations])
                    ].copy()

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
                    split_mode=split_mode,
                    run_dir=run_dir,
                    export_per_location_files=bool(export_per_location_files),
                )

                summary_df = pd.DataFrame([summary])

                train_counts = (
                    df.loc[df["location_key"].astype(str).isin([str(x) for x in selected_locations]), "location_key"]
                    .astype(str)
                    .value_counts()
                    .rename_axis("location_key")
                    .reset_index(name="train_source_rows")
                )
                if forecast_base_df is not None and not forecast_base_df.empty:
                    test_counts = (
                        forecast_base_df["location_key"].astype(str).value_counts().rename_axis("location_key").reset_index(name="test_source_rows")
                    )
                else:
                    test_counts = pd.DataFrame({"location_key": selected_locations, "test_source_rows": [0] * len(selected_locations)})

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

        st.write("### Dự báo 24 giờ tiếp theo (từng địa điểm)")
        st.dataframe(future_df.head(300), use_container_width=True)
        st.download_button(
            "Download file tổng (mọi location)",
            data=future_df.to_csv(index=False).encode("utf-8"),
            file_name="future_24h_predictions.csv",
            mime="text/csv",
        )

        if export_per_location_files:
            st.info("Đã xuất thêm file riêng cho từng location trong thư mục run, ví dụ: future_24h_predictions_hcm.csv")
        else:
            st.info("Đang dùng chế độ file tổng: future_24h_predictions.csv (có cột location_key).")
        if summary.get("per_location_files"):
            st.write("### Các file đã xuất")
            st.dataframe(pd.DataFrame({"file": summary["per_location_files"]}), use_container_width=True)

        st.code(
            "\n".join(
                [
                    f"run_dir: {os.path.dirname(summary['future_pred_path'])}",
                    "Files: future_24h_predictions_<location>.csv",
                ]
            )
        )


if __name__ == "__main__":
    main()

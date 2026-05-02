import argparse, os, json, glob, time
import pandas as pd, numpy as np
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from utils import make_windows_grouped, make_windows_grouped_with_locids, scale_features, rmse, mae, mape
import joblib


# ─────────────────────────────────────────────
#  Model
# ─────────────────────────────────────────────

class LSTMForecaster(nn.Module):
    """
    LSTM Forecaster with optional Location Embedding.
    - num_locations > 1  → dùng nn.Embedding để mã hoá tỉnh thành
    - num_locations <= 1 → không dùng embedding (single-station mode)

    Batch-prediction mode: horizon = forecast_steps (ví dụ 24).
    Model xuất thẳng (B, horizon) — không cần autoregressive loop.
    """
    def __init__(self, input_size=1, hidden_size=256, num_layers=4, dropout=0.3,
                 horizon=24, num_locations=0, embed_dim=32):
        super().__init__()
        self.use_embedding = num_locations > 1
        if self.use_embedding:
            self.loc_embedding = nn.Embedding(num_locations, embed_dim)
            lstm_input = input_size + embed_dim
        else:
            lstm_input = input_size

        # Multi-layer projection before LSTM
        self.input_proj = nn.Sequential(
            nn.Linear(lstm_input, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
        )

        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0.0)

        # Multi-layer output projection → xuất thẳng horizon bước
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, horizon),
        )

    def forward(self, x, loc_id=None):
        """
        x      : (B, T, input_size)
        loc_id : (B,) int tensor — chỉ cần khi use_embedding=True
        return : (B, horizon)
        """
        if self.use_embedding and loc_id is not None:
            loc_emb = self.loc_embedding(loc_id)                       # (B, embed_dim)
            loc_emb = loc_emb.unsqueeze(1).expand(-1, x.size(1), -1)  # (B, T, embed_dim)
            x = torch.cat([x, loc_emb], dim=-1)                        # (B, T, input+embed)
        x = self.input_proj(x)      # (B, T, hidden_size)
        out, _ = self.lstm(x)       # (B, T, hidden_size)
        h_last = out[:, -1, :]      # (B, hidden_size)
        return self.output_proj(h_last)  # (B, horizon)


# ─────────────────────────────────────────────
#  Plot helpers
# ─────────────────────────────────────────────

def plot_curves(history, outpath):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(history["train_loss"], label="train_loss")
    ax.plot(history["val_loss"],   label="val_loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE)")
    ax.set_title("Training & Validation Loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)


def plot_forecast(dates, y_true, preds, outpath):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(dates, y_true, label="actual")
    ax.plot(dates, preds,  label="forecast")
    ax.set_title("Forecast vs Actual")
    ax.set_xlabel("Time (ts_utc)")
    ax.set_ylabel("Value")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)


# ─────────────────────────────────────────────
#  Train & Eval Helpers
# ─────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, dataloader, crit, device, use_embedding, amp_enabled):
    model.eval()
    vloss, vn = 0.0, 0
    all_vpreds, all_vtrues = [], []
    for xb, lid, yb in tqdm(dataloader, desc="[val]", leave=False):
        xb, lid, yb = xb.to(device, non_blocking=True), lid.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            preds = model(xb, lid if use_embedding else None)
            loss  = crit(preds, yb)
        vloss += loss.item() * xb.size(0)
        vn    += xb.size(0)
        all_vpreds.append(preds.detach().float().cpu().numpy())
        all_vtrues.append(yb.detach().float().cpu().numpy())
    return vloss / max(vn, 1), all_vpreds, all_vtrues

def run_epoch(model, dataloader, opt, crit, scaler_amp, device, use_embedding, grad_accum_steps, max_grad_norm, amp_enabled, ep, epochs):
    model.train()
    tloss, n = 0.0, 0
    opt.zero_grad(set_to_none=True)
    for step, (xb, lid, yb) in enumerate(tqdm(dataloader, desc=f"Epoch {ep}/{epochs} [train]")):
        xb, lid, yb = xb.to(device, non_blocking=True), lid.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            preds = model(xb, lid if use_embedding else None)
            loss  = crit(preds, yb) / grad_accum_steps
            
        if amp_enabled:
            scaler_amp.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(dataloader):
            if amp_enabled:
                scaler_amp.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            if amp_enabled:
                scaler_amp.step(opt)
                scaler_amp.update()
            else:
                opt.step()
            opt.zero_grad(set_to_none=True)

        tloss += loss.item() * grad_accum_steps * xb.size(0)
        n     += xb.size(0)
    return tloss / max(n, 1)


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir",         type=str,   default="data")
    ap.add_argument("--target",           type=str,   default="aqi")
    ap.add_argument("--location",         type=str,   default="all")
    ap.add_argument("--lookback",         type=int,   default=24)
    # [CHANGE ②] horizon = số bước dự báo batch (thay thế forecast-steps)
    ap.add_argument("--horizon",          type=int,   default=24,
                    help="Số bước dự báo. Model xuất thẳng (B, horizon) — batch prediction.")
    ap.add_argument("--epochs",           type=int,   default=30)
    ap.add_argument("--batch-size",       type=int,   default=512)
    ap.add_argument("--lr",               type=float, default=3e-4)
    ap.add_argument("--weight-decay",     type=float, default=1e-4)
    ap.add_argument("--grad-accum-steps", type=int,   default=2)
    ap.add_argument("--max-grad-norm",    type=float, default=1.0)
    ap.add_argument("--outdir",           type=str,   default="outputs")
    ap.add_argument("--seed",             type=int,   default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    # ── 1. Đọc dữ liệu ──────────────────────────────────────────────────────
    csv_files = glob.glob(os.path.join(args.data_dir, "*.csv"))
    dfs = []
    for file in csv_files:
        temp_df = pd.read_csv(file)
        if "location_key" in temp_df.columns and "ts_utc" in temp_df.columns:
            temp_df["ts_utc"] = pd.to_datetime(temp_df["ts_utc"])
            dfs.append(temp_df)

    if not dfs:
        print("[ERROR] No valid CSV files with 'ts_utc' and 'location_key' found.")
        return

    df = pd.concat(dfs, ignore_index=True)
    df.sort_values(by=["location_key", "ts_utc"], inplace=True)

    if args.location and args.location.lower() != "all":
        df = df[df["location_key"] == args.location]
        if len(df) == 0:
            print(f"[ERROR] No data found for location_key: {args.location}")
            return
        args.outdir = os.path.join(args.outdir, args.location)
    else:
        print("[INFO] Training on ALL locations globally (Global Model).")
        args.outdir = os.path.join(args.outdir, "global_model")

    os.makedirs(args.outdir, exist_ok=True)

    # ── 2. Xác định feature columns ─────────────────────────────────────────
    exclude_cols = ["ts_utc", "location_key", "date"]
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in exclude_cols]

    if args.target not in feature_cols:
        print(f"[ERROR] Target column '{args.target}' not found in numeric columns.")
        return

    # ── [CHANGE ③] Data Preprocessing: log shape + ffill trước dropna ───────
    print(f"[INFO] Raw data shape  : {df.shape}")
    null_counts = df[feature_cols].isnull().sum()
    null_counts = null_counts[null_counts > 0]
    if len(null_counts):
        print(f"[INFO] Null counts (trước fill):\n{null_counts.to_string()}")

    # ffill theo từng location, sau đó bfill để lấp phần đầu chuỗi
    df[feature_cols] = df.groupby("location_key")[feature_cols].transform(
        lambda x: x.ffill().bfill()
    )
    df.dropna(subset=feature_cols, inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"[INFO] After cleaning   : {df.shape}")
    print(f"[INFO] Features ({len(feature_cols)}): {feature_cols}")
    print(f"[INFO] Target           : {args.target}")

    # ── 3. Location embedding mapping ───────────────────────────────────────
    all_locations = sorted(df["location_key"].unique().tolist())
    num_locations = len(all_locations)
    loc2id        = {loc: idx for idx, loc in enumerate(all_locations)}
    use_embedding = num_locations > 1
    if use_embedding:
        print(f"[INFO] {num_locations} locations → dùng Location Embedding (embed_dim=32).")
    else:
        print(f"[INFO] 1 location → không dùng Embedding.")

    # ── 4. Sliding windows (Không scale trước) ────────────────────────────────
    print("[INFO] Creating windows...")
    if use_embedding:
        X, y, loc_ids = make_windows_grouped_with_locids(
            df, feature_cols, args.target, args.lookback, args.horizon,
            loc2id, group_col="location_key"
        )
    else:
        X, y = make_windows_grouped(
            df, feature_cols, args.target, args.lookback, args.horizon,
            group_col="location_key"
        )
        loc_ids = np.zeros(len(X), dtype=np.int64)

    if len(X) == 0:
        print("[ERROR] Not enough data to create windows.")
        return

    print(f"[INFO] Window shapes — X: {X.shape}, y: {y.shape}")

    X       = X.astype(np.float32)
    y       = y.astype(np.float32)
    loc_ids = loc_ids.astype(np.int64)

    # ── 5. Train / Val / Test split (70 / 10 / 20) ──────────────────────────
    total_len = len(X)
    train_idx = int(0.7 * total_len)
    val_idx   = int(0.8 * total_len)

    # ── 6. Normalize (Fit on Train, Apply to All) ───────────────────────────
    x_mean = X[:train_idx].mean(axis=(0, 1), keepdims=True)
    x_std  = X[:train_idx].std(axis=(0, 1),  keepdims=True)
    x_std  = np.where(x_std < 1e-6, 1.0, x_std)
    X      = (X - x_mean) / x_std

    y_mean = float(y[:train_idx].mean())
    y_std  = float(y[:train_idx].std())
    if y_std < 1e-6:
        y_std = 1.0
    y      = (y - y_mean) / y_std

    X_train, y_train = X[:train_idx],        y[:train_idx]
    X_val,   y_val   = X[train_idx:val_idx], y[train_idx:val_idx]
    X_test,  y_test  = X[val_idx:],          y[val_idx:]
    lid_train        = loc_ids[:train_idx]
    lid_val          = loc_ids[train_idx:val_idx]
    lid_test         = loc_ids[val_idx:]

    tr_ds = TensorDataset(torch.tensor(X_train), torch.tensor(lid_train), torch.tensor(y_train))
    va_ds = TensorDataset(torch.tensor(X_val),   torch.tensor(lid_val),   torch.tensor(y_val))
    te_ds = TensorDataset(torch.tensor(X_test),  torch.tensor(lid_test),  torch.tensor(y_test))

    tr_dl = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    va_dl = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    te_dl = DataLoader(te_ds, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    print(f"[INFO] Split — Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # ── 7. Model / Optimizer / AMP ───────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda"
    print(f"[INFO] Device: {device}, AMP enabled: {amp_enabled}")

    model = LSTMForecaster(
        input_size    = len(feature_cols),
        horizon       = args.horizon,       # batch output size
        num_locations = num_locations,
        embed_dim     = 32,
    ).to(device)

    opt    = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    crit   = nn.HuberLoss(delta=1.0)
    scaler_amp = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    best_val  = float("inf")
    stale     = 0
    best_path = os.path.join(args.outdir, "best_lstm.pt")

    # [CHANGE ① + ④] history có train_sec, val_sec, run_sec riêng biệt
    history = {
        "epoch":      [],
        "train_loss": [], "val_loss": [],
        "val_rmse":   [], "val_mae": [], "val_r2": [],
        "train_sec":  [],   # ← thời gian chỉ tính forward/backward
        "val_sec":    [],   # ← thời gian eval riêng
        "run_sec":    [],   # ← tổng epoch (train + val)
    }

    # ── 8. Training Loop ─────────────────────────────────────────────────────
    run_start = time.time()

    for ep in range(1, args.epochs + 1):

        # Train
        train_start = time.time()
        tl = run_epoch(
            model, tr_dl, opt, crit, scaler_amp, device, use_embedding, 
            args.grad_accum_steps, args.max_grad_norm, amp_enabled, ep, args.epochs
        )
        train_sec = time.time() - train_start

        # Val
        val_start = time.time()
        vl, all_vpreds, all_vtrues = evaluate(
            model, va_dl, crit, device, use_embedding, amp_enabled
        )
        val_sec = time.time() - val_start
        run_sec = train_sec + val_sec

        # Inverse-transform để tính metrics thực
        vpred_mat  = np.concatenate(all_vpreds, axis=0) * y_std + y_mean
        vtrue_mat  = np.concatenate(all_vtrues, axis=0) * y_std + y_mean
        
        v_rmse = rmse(vtrue_mat, vpred_mat)
        v_mae  = mae(vtrue_mat, vpred_mat)
        v_r2   = r2_score(vtrue_mat.flatten(), vpred_mat.flatten())

        # [CHANGE ④] Log đầy đủ 3 cột thời gian — giống Mamba log format
        print(
            f"[Epoch {ep:02d}] "
            f"train_loss={tl:.4f} val_loss={vl:.4f} | "
            f"val_rmse={v_rmse:.2f} val_mae={v_mae:.2f} val_r2={v_r2:.4f} | "
            f"train={train_sec:.1f}s val={val_sec:.1f}s run={run_sec:.1f}s"
        )

        history["epoch"].append(ep)
        history["train_loss"].append(tl)
        history["val_loss"].append(vl)
        history["val_rmse"].append(v_rmse)
        history["val_mae"].append(v_mae)
        history["val_r2"].append(v_r2)
        history["train_sec"].append(train_sec)
        history["val_sec"].append(val_sec)
        history["run_sec"].append(run_sec)

        # [CHANGE ④] Checkpoint log rõ ràng — giống Mamba
        if vl < best_val:
            best_val = vl
            stale    = 0
            torch.save({
                "model_state":   model.state_dict(),
                "input_size":    len(feature_cols),
                "hidden_size":   256,
                "num_layers":    4,
                "embed_dim":     32,
                "dropout":       0.3,
                "horizon":       args.horizon,
                "lookback":      args.lookback,
                "feature_cols":  feature_cols,
                "x_mean":        x_mean,
                "x_std":         x_std,
                "y_mean":        y_mean,
                "y_std":         y_std,
                "use_embedding": use_embedding,
                "num_locations": num_locations,
                "loc2id":        loc2id,
                # [CHANGE ④] metadata thêm vào checkpoint
                "best_epoch":    ep,
                "best_val_loss": vl,
                "best_val_rmse": v_rmse,
            }, best_path)
            print(f"[CHECKPOINT] Saved best model → epoch={ep}, val_loss={vl:.4f}, path={best_path}")
        else:
            stale += 1
            if stale >= 5:
                print("[INFO] Early stopping triggered.")
                break

    total_train_time = time.time() - run_start

    # Lưu epoch metrics — epoch_metrics.csv có đủ 3 cột thời gian
    metrics_path = os.path.join(args.outdir, "epoch_metrics.csv")
    pd.DataFrame(history).to_csv(metrics_path, index=False)
    plot_curves(history, os.path.join(args.outdir, "loss_curve.png"))
    print(f"[OK] Training complete in {total_train_time:.1f}s total.")
    print(f"[OK] Epoch metrics saved → {metrics_path}")

    # ── 9. Load best model ───────────────────────────────────────────────────
    state = torch.load(best_path, map_location=device)
    model.load_state_dict(state["model_state"])
    model.to(device)
    model.eval()

    # ── 10. [CHANGE ②] Batch Prediction — 1 lần forward, không autoregressive ──
    print(f"\n[INFO] Batch prediction — horizon={args.horizon} bước, 1 lần forward/location.")
    all_future_dfs = []

    x_mean_1d = x_mean.squeeze()
    x_std_1d  = x_std.squeeze()

    for loc_key in df["location_key"].unique():
        loc_group = df[df["location_key"] == loc_key].sort_values("ts_utc")

        if len(loc_group) < args.lookback:
            print(f"[WARNING] {loc_key}: không đủ dữ liệu lookback={args.lookback}, bỏ qua.")
            continue

        # Lấy cửa sổ lookback giờ cuối cùng
        last_window = loc_group[feature_cols].iloc[-args.lookback:].values.astype("float32")
        last_window_scaled = (last_window - x_mean_1d) / x_std_1d
        x_input     = torch.tensor(last_window_scaled).unsqueeze(0).to(device)  # (1, lookback, F)

        if use_embedding:
            loc_id_tensor = torch.tensor([loc2id[loc_key]], dtype=torch.long).to(device)
        else:
            loc_id_tensor = None

        # [CHANGE ②] 1 lần forward → (1, horizon)
        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                pred_scaled = model(x_input, loc_id_tensor).cpu().float().numpy().flatten()  # (horizon,)

        # Inverse transform
        future_pred_orig = pred_scaled * y_std + y_mean

        # Tạo timestamps
        last_date      = loc_group["ts_utc"].iloc[-1]
        freq           = pd.Timedelta(hours=1)
        if len(loc_group["ts_utc"]) > 1:
            freq = loc_group["ts_utc"].diff().mode().iloc[0]
        next_day_start = (last_date + pd.Timedelta(days=1)).normalize()
        future_dates   = pd.date_range(start=next_day_start, periods=args.horizon, freq=freq)

        f_df = pd.DataFrame({
            "location_key":            loc_key,
            "ts_utc":                  future_dates,
            f"forecast_{args.target}": future_pred_orig,
        })
        all_future_dfs.append(f_df)
        print(f"[INFO] {loc_key}: {args.horizon} bước dự báo (batch, 1 forward pass).")

    if all_future_dfs:
        future_df         = pd.concat(all_future_dfs, ignore_index=True)
        forecast_csv_path = os.path.join(args.outdir, "future_forecast.csv")
        future_df.to_csv(forecast_csv_path, index=False)
        print(f"[OK] Forecast saved → {forecast_csv_path} ({len(future_df)} rows)")
    else:
        print("[WARNING] Không có đủ dữ liệu để tạo forecast.")


if __name__ == "__main__":
    main()
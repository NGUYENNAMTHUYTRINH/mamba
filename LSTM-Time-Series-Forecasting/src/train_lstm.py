import argparse, os, json, glob, time
import pandas as pd, numpy as np
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from utils import make_windows_grouped, make_windows_grouped_with_locids, scale_features, rmse, mae, mape
import joblib

class LSTMForecaster(nn.Module):
    """
    LSTM Forecaster with optional Location Embedding.
    - num_locations > 1  → dùng nn.Embedding để mã hoá tỉnh thành
    - num_locations <= 1 → không dùng embedding (single-station mode)
    """
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2,
                 horizon=1, num_locations=0, embed_dim=8):
        super().__init__()
        self.use_embedding = num_locations > 1
        if self.use_embedding:
            self.loc_embedding = nn.Embedding(num_locations, embed_dim)
            lstm_input = input_size + embed_dim
        else:
            lstm_input = input_size
        self.lstm = nn.LSTM(lstm_input, hidden_size, num_layers=num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, horizon)

    def forward(self, x, loc_id=None):
        """
        x      : (B, T, input_size)
        loc_id : (B,) int tensor  — chỉ cần thiết khi use_embedding=True
        """
        if self.use_embedding and loc_id is not None:
            loc_emb = self.loc_embedding(loc_id)                      # (B, embed_dim)
            loc_emb = loc_emb.unsqueeze(1).expand(-1, x.size(1), -1) # (B, T, embed_dim)
            x = torch.cat([x, loc_emb], dim=-1)                       # (B, T, input+embed)
        out, _ = self.lstm(x)
        h_last = out[:, -1, :]   # (B, hidden_size)
        return self.fc(h_last)

def plot_curves(history, outpath):
    fig, ax = plt.subplots(figsize=(7,5))
    ax.plot(history["train_loss"], label="train_loss")
    ax.plot(history["val_loss"], label="val_loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE)")
    ax.set_title("Training & Validation Loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)

def plot_forecast(dates, y_true, preds, outpath):
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(dates, y_true, label="actual")
    ax.plot(dates, preds, label="forecast")
    ax.set_title("Forecast vs Actual")
    ax.set_xlabel("Time (ts_utc)")
    ax.set_ylabel("Value")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, default="data", help="Directory containing input data files (e.g. 2025.csv)")
    ap.add_argument("--target", type=str, default="aqi", help="Target column to predict")
    ap.add_argument("--location", type=str, default="all", help="Filter by a specific location_key. Use 'all' to train a global model on all locations.")
    ap.add_argument("--lookback", type=int, default=24, help="Number of time steps to look back (e.g., 24 hours)")
    ap.add_argument("--horizon", type=int, default=1, help="Number of time steps to predict forward per step (should be 1 for autoregressive mode)")
    ap.add_argument("--forecast-steps", type=int, default=24, help="Number of future hours to forecast autoregressively")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--outdir", type=str, default="outputs")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    # 1. Read all CSV files in data directory
    csv_files = glob.glob(os.path.join(args.data_dir, "*.csv"))
    dfs = []
    
    for file in csv_files:
        temp_df = pd.read_csv(file)
        # Verify it has necessary multivariate structure format (location_key, ts_utc)
        if "location_key" in temp_df.columns and "ts_utc" in temp_df.columns:
            # Parse dates
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
            
    print(f"[INFO] Filtered DataFrame with {len(df)} rows.")
    
    # Fill missing values dynamically or drop
    df.dropna(inplace=True)

    # --- Location ID mapping (dùng cho Embedding khi có nhiều tỉnh) ---
    all_locations = sorted(df["location_key"].unique().tolist())
    num_locations = len(all_locations)
    loc2id = {loc: idx for idx, loc in enumerate(all_locations)}  # {"hanoi": 0, "hcm": 1, ...}
    use_embedding = num_locations > 1
    if use_embedding:
        print(f"[INFO] Detected {num_locations} locations => su dung Location Embedding (embed_dim=8).")
    else:
        print(f"[INFO] Only 1 location detected => khong dung Embedding.")

    # 2. Automatically find numeric features
    exclude_cols = ['ts_utc', 'location_key', 'date']
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in exclude_cols]
    
    if args.target not in feature_cols:
        print(f"[ERROR] Target column {args.target} not found in numeric columns.")
        return
        
    print(f"[INFO] Features count: {len(feature_cols)}. Target: {args.target}")

    # 3. Scale Features and Target
    df_scaled, X_scaler, y_scaler = scale_features(df.copy(), feature_cols, args.target)

    # 4. Make Windows per Location to avoid mixing sequences
    print("[INFO] Creating windows...")
    if use_embedding:
        X, y, loc_ids = make_windows_grouped_with_locids(
            df_scaled, feature_cols, args.target, args.lookback, args.horizon,
            loc2id, group_col="location_key"
        )
    else:
        X, y = make_windows_grouped(
            df_scaled, feature_cols, args.target, args.lookback, args.horizon,
            group_col="location_key"
        )
        loc_ids = np.zeros(len(X), dtype=np.int64)  # dummy, không dùng
    
    if len(X) == 0:
        print("[ERROR] Not enough data to create windows for given lookback/horizon.")
        return

    print(f"[INFO] Window shapes - X: {X.shape}, y: {y.shape}")

    # Convert to float32 / int64
    X        = X.astype(np.float32)
    y        = y.astype(np.float32)
    loc_ids  = loc_ids.astype(np.int64)

    # 5. Train/Val/Test Split (70/10/20)
    total_len = len(X)
    train_idx = int(0.7 * total_len)
    val_idx   = int(0.8 * total_len)
    
    X_train,    y_train    = X[:train_idx],        y[:train_idx]
    X_val,      y_val      = X[train_idx:val_idx], y[train_idx:val_idx]
    X_test,     y_test     = X[val_idx:],          y[val_idx:]
    lid_train              = loc_ids[:train_idx]
    lid_val                = loc_ids[train_idx:val_idx]
    lid_test               = loc_ids[val_idx:]

    tr_ds = TensorDataset(torch.tensor(X_train), torch.tensor(lid_train), torch.tensor(y_train))
    va_ds = TensorDataset(torch.tensor(X_val),   torch.tensor(lid_val),   torch.tensor(y_val))
    te_ds = TensorDataset(torch.tensor(X_test),  torch.tensor(lid_test),  torch.tensor(y_test))
    
    tr_dl = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True)
    va_dl = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False)
    te_dl = DataLoader(te_ds, batch_size=args.batch_size, shuffle=False)
    
    print(f"[INFO] Split sizes - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # 6. Model definition
    model = LSTMForecaster(
        input_size=len(feature_cols),
        horizon=args.horizon,
        num_locations=num_locations,
        embed_dim=8
    )
    opt  = torch.optim.Adam(model.parameters(), lr=args.lr)
    crit = nn.MSELoss()

    best_val = float("inf")
    stale = 0
    history = {"epoch": [], "train_loss": [], "val_loss": [], "val_rmse": [], "val_mae": [], "val_r2": [], "time_sec": []}
    best_path = os.path.join(args.outdir, "best_lstm.pt")

    # 7. Training Loop
    for ep in range(1, args.epochs + 1):
        start_t = time.time()
        
        model.train()
        tloss = 0.0
        n = 0
        for xb, lid, yb in tqdm(tr_dl, desc=f"Epoch {ep}/{args.epochs} [train]"):
            opt.zero_grad()
            preds = model(xb, lid if use_embedding else None)
            loss  = crit(preds, yb)
            loss.backward()
            opt.step()
            tloss += loss.item() * xb.size(0)
            n     += xb.size(0)
        tl = tloss / n
        
        model.eval()
        vloss = 0.0
        vn = 0
        all_vpreds, all_vtrues = [], []
        with torch.no_grad():
            for xb, lid, yb in tqdm(va_dl, desc=f"Epoch {ep}/{args.epochs} [val]"):
                preds = model(xb, lid if use_embedding else None)
                loss  = crit(preds, yb)
                vloss += loss.item() * xb.size(0)
                vn    += xb.size(0)
                all_vpreds.append(preds.numpy())
                all_vtrues.append(yb.numpy())
        vl = vloss / vn
        
        # Calculate real unscaled metrics for val
        vpred_mat = np.concatenate(all_vpreds, axis=0)
        vtrue_mat = np.concatenate(all_vtrues, axis=0)
        orig_s = vpred_mat.shape
        vpred_orig = y_scaler.inverse_transform(vpred_mat.reshape(-1, 1)).reshape(orig_s)
        vtrue_orig = y_scaler.inverse_transform(vtrue_mat.reshape(-1, 1)).reshape(orig_s)
        
        v_rmse = rmse(vtrue_orig, vpred_orig)
        v_mae = mae(vtrue_orig, vpred_orig)
        v_r2 = r2_score(vtrue_orig.flatten(), vpred_orig.flatten())
        
        elapsed = time.time() - start_t
        
        history["epoch"].append(ep)
        history["train_loss"].append(tl)
        history["val_loss"].append(vl)
        history["val_rmse"].append(v_rmse)
        history["val_mae"].append(v_mae)
        history["val_r2"].append(v_r2)
        history["time_sec"].append(elapsed)
        print(f"[Epoch {ep}] train_loss={tl:.4f} val_loss={vl:.4f} | val_rmse={v_rmse:.2f} val_mae={v_mae:.2f} val_r2={v_r2:.4f} | {elapsed:.1f}s")
        
        if vl < best_val:
            best_val = vl
            stale = 0
            torch.save({
                "model_state":  model.state_dict(),
                "input_size":   len(feature_cols),
                "horizon":      args.horizon,
                "lookback":     args.lookback,
                "feature_cols": feature_cols,
                "X_scaler":     X_scaler,
                "y_scaler":     y_scaler,
                # --- Embedding metadata ---
                "use_embedding":  use_embedding,
                "num_locations":  num_locations,
                "loc2id":         loc2id,
            }, best_path)
        else:
            stale += 1
            if stale >= 5:
                print("Early stopping triggered.")
                break

    # Training complete
    
    # Save training history metric
    pd.DataFrame(history).to_csv(os.path.join(args.outdir, "epoch_metrics.csv"), index=False)
    print("[OK] Training complete. Epoch Metrics saved.")

    state = torch.load(best_path, map_location="cpu")
    model.load_state_dict(state["model_state"])
    model.eval()

    # 9. Future Forecast — Autoregressive (Rolling) Decoding
    # Mỗi bước: dự đoán 1 giờ tiếp theo → cập nhật cửa sổ → lặp lại
    all_future_dfs = []
    target_col_idx = feature_cols.index(args.target)  # vị trí cột AQI trong feature_cols

    for loc_key in df_scaled["location_key"].unique():
        loc_group = df_scaled[df_scaled["location_key"] == loc_key].sort_values("ts_utc")
        if len(loc_group) < args.lookback:
            print(f"[WARNING] {loc_key}: không đủ dữ liệu cho lookback={args.lookback}, bỏ qua.")
            continue

        # Cửa sổ ban đầu: lookback giờ cuối cùng (scaled)
        rolling_window = loc_group[feature_cols].iloc[-args.lookback:].values.copy().astype("float32")

        # Tần suất & mốc thời gian
        last_date = loc_group["ts_utc"].iloc[-1]
        freq = pd.Timedelta(hours=1)
        if len(loc_group["ts_utc"]) > 1:
            freq = loc_group["ts_utc"].diff().mode().iloc[0]

        # Snap về 0h ngày hôm sau
        next_day_start = (last_date + pd.Timedelta(days=1)).normalize()

        if use_embedding:
            loc_id_tensor = torch.tensor([loc2id[loc_key]], dtype=torch.long)
        else:
            loc_id_tensor = None

        # Autoregressive loop: dự đoán từng bước một
        future_preds_scaled = []
        for step in range(args.forecast_steps):
            x_input = torch.tensor(rolling_window).unsqueeze(0)  # (1, lookback, features)
            with torch.no_grad():
                pred_scaled = model(x_input, loc_id_tensor).numpy().flatten()  # (horizon,)

            # Chỉ lấy bước đầu tiên (horizon=1 hoặc dùng pred[0] khi horizon>1)
            next_aqi_scaled = float(pred_scaled[0])
            future_preds_scaled.append(next_aqi_scaled)

            # Cập nhật cửa sổ: tạo hàng mới = copy hàng cuối, thay AQI bằng giá trị vừa đoán
            new_row = rolling_window[-1].copy()
            new_row[target_col_idx] = next_aqi_scaled

            # Trượt cửa sổ: bỏ hàng cũ nhất, thêm hàng mới
            rolling_window = np.vstack([rolling_window[1:], new_row])

        # Inverse transform về giá trị AQI gốc
        future_pred_orig = y_scaler.inverse_transform(
            np.array(future_preds_scaled).reshape(-1, 1)
        ).flatten()

        future_dates = pd.date_range(start=next_day_start, periods=args.forecast_steps, freq=freq)

        f_df = pd.DataFrame({
            "location_key": loc_key,
            "ts_utc": future_dates,
            f"forecast_{args.target}": future_pred_orig
        })
        all_future_dfs.append(f_df)

    if len(all_future_dfs) > 0:
        future_df = pd.concat(all_future_dfs, ignore_index=True)
        tomorrow_csv_path = os.path.join(args.outdir, "future_forecast.csv")
        future_df.to_csv(tomorrow_csv_path, index=False)
        print(f"[INFO] Extracted {len(future_df)} forecast points (autoregressive, {args.forecast_steps} steps).")
        print(f"[INFO] Saved to {tomorrow_csv_path}")
    else:
        print("[WARNING] Not enough data to yield a forecast.")

if __name__ == "__main__":
    main()

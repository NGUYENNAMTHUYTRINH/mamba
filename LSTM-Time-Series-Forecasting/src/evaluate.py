import argparse, os, json, glob
import pandas as pd, numpy as np, torch
from utils import make_windows_grouped, rmse, mae, mape
import joblib
import matplotlib.pyplot as plt
import torch.nn as nn

class LSTMForecaster(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2, horizon=30):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, horizon)
    def forward(self, x):
        out, _ = self.lstm(x)
        h_last = out[:, -1, :]
        return self.fc(h_last)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", type=str, default="dataset/2025.csv", help="Path to input data file (e.g. 2025.csv)")
    ap.add_argument("--target", type=str, default="aqi", help="Target column to predict")
    ap.add_argument("--location", type=str, default=None, help="Filter by a specific location_key")
    ap.add_argument("--model", type=str, default="outputs/best_lstm.pt")
    ap.add_argument("--lookback", type=int, default=60)
    ap.add_argument("--horizon", type=int, default=30)
    ap.add_argument("--outdir", type=str, default="outputs")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print(f"[INFO] Loading dataset from: {args.data_path}")
    try:
        df = pd.read_csv(args.data_path)
        if "location_key" in df.columns and "ts_utc" in df.columns:
            df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True, errors="coerce")
            df["ts_utc"] = df["ts_utc"].dt.tz_localize(None) 
        else:
            print("[ERROR] CSV file missing 'ts_utc' or 'location_key'.")
            return
    except Exception as e:
        print(f"[ERROR] Could not read {args.data_path}: {e}")
        return

    df.sort_values(by=["location_key", "ts_utc"], inplace=True)
    
    if not args.location:
        args.location = df["location_key"].unique()[0]
        print(f"[INFO] Auto-selected location_key: {args.location}")
        
    df = df[df["location_key"] == args.location]
    if len(df) == 0:
        print(f"[ERROR] No data found for location_key: {args.location}")
        return

    args.outdir = os.path.join(args.outdir, args.location)
    os.makedirs(args.outdir, exist_ok=True)
    
    # Defaults usually passed as outputs/best_lstm.pt string, override if unchanged.
    if args.model == "outputs/best_lstm.pt":
        args.model = os.path.join(args.outdir, "best_lstm.pt")
            
    df.dropna(inplace=True)

    exclude_cols = ['ts_utc', 'location_key', 'date']
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in exclude_cols]
    
    if args.target not in feature_cols:
        print(f"[ERROR] Target column {args.target} not found in numeric columns.")
        return

    # Load Model state to get proper dimension scaling and scalers
    state = torch.load(args.model, map_location="cpu")
    input_size = state.get("input_size", len(feature_cols))
    
    X_scaler = state["X_scaler"]
    y_scaler = state["y_scaler"]
    
    # Scale Data
    df_scaled = df.copy()
    df_scaled[feature_cols] = X_scaler.transform(df_scaled[feature_cols])
    df_scaled[args.target] = y_scaler.transform(df_scaled[[args.target]])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Evaluating on device: {device}")
    
    model = LSTMForecaster(input_size=input_size, horizon=args.horizon)
    model.load_state_dict(state["model_state"] if "model_state" in state else state)
    model.to(device)
    model.eval()

    # Create windows for entire set
    X, y = make_windows_grouped(df_scaled, feature_cols, args.target, args.lookback, args.horizon)
    X = X.astype(np.float32)

    # Evaluate visually on the last sequence from the last location
    last_loc = df_scaled["location_key"].iloc[-1]
    last_group = df_scaled[df_scaled["location_key"] == last_loc]
    
    if len(last_group) > args.lookback + args.horizon:
        dates = last_group["ts_utc"].iloc[-args.horizon:].values
        
        feature_slice = last_group[feature_cols].iloc[-args.horizon-args.lookback : -args.horizon]
        last_input = torch.tensor(feature_slice.values.astype("float32")).unsqueeze(0).to(device)
        
        with torch.no_grad():
            pred_scaled = model(last_input).cpu().numpy().flatten()
            
        pred = y_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
        y_true = last_group[args.target].iloc[-args.horizon:].values
        y_true_orig = y_scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()
        
        r = pd.DataFrame([{
            "rmse": rmse(y_true_orig, pred), 
            "mae": mae(y_true_orig, pred), 
            "mape": mape(y_true_orig, pred)
        }])
        r.to_csv(os.path.join(args.outdir, "evaluate_metrics.csv"), index=False)
        
        print("[OK] Evaluation complete. Evaluate metrics saved to CSV.")
        
        # 3. Predict into the future (e.g. next 24h)
        future_slice = last_group[feature_cols].iloc[-args.lookback:]
        future_input = torch.tensor(future_slice.values.astype("float32")).unsqueeze(0).to(device)
        
        with torch.no_grad():
            future_pred_scaled = model(future_input).cpu().numpy().flatten()
        future_pred = y_scaler.inverse_transform(future_pred_scaled.reshape(-1, 1)).flatten()
        
        # Generate future dates
        last_date = last_group["ts_utc"].iloc[-1]
        freq = last_group["ts_utc"].diff().mode().iloc[0] # automatically find frequency
        future_dates = pd.date_range(start=last_date + freq, periods=args.horizon, freq=freq)
        
        future_df = pd.DataFrame({
            "ts_utc": future_dates,
            f"forecast_{args.target}": future_pred
        })
        
        # --- Extract Tomorrow Only ---
        tomorrow_date = (last_date + pd.Timedelta(days=1)).date()
        tomorrow_df = future_df[future_df["ts_utc"].dt.date == tomorrow_date]
        
        if len(tomorrow_df) > 0:
            tomorrow_df = tomorrow_df.copy()
            tomorrow_df.insert(0, "location_key", args.location)
            tomorrow_csv_path = os.path.join(args.outdir, "future_forecast.csv")
            tomorrow_df.to_csv(tomorrow_csv_path, index=False)
            print(f"[INFO] Future forecast for Tomorrow ({tomorrow_date}) saved to {tomorrow_csv_path}")
        else:
            print(f"[WARNING] Horizon ({args.horizon}) is too short to reach tomorrow ({tomorrow_date}). Please use --horizon 48")

    else:
        print("[ERROR] Last location sequence too short for evaluate plotting.")

if __name__ == "__main__":
    main()

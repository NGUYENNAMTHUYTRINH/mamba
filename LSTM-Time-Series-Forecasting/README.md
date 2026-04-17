# LSTM Time-Series Forecasting
A hands-on project for forecasting time-series with PyTorch LSTMs. It creates realistic daily data (trend, seasonality, events, noise), prepares it with sliding windows, and trains an LSTM to make multi-step predictions. The project tracks errors with RMSE, MAE, MAPE and shows clear plots of training progress and forecast results.

---

## Features
- Synthetic daily series generation (configurable length & seed)
- Sliding-window dataset preparation for supervised learning
- LSTM model with dropout, Adam optimizer, early stopping
- Multi-step forecasting (direct & recursive)
- Metrics: RMSE, MAE, MAPE
- Visualizations: training curves & forecast plots
- Saved artifacts: `best_lstm.pt`, `scaler.pkl`, `metrics.json`

## Results
- **RMSE:** 22.25  
- **MAE:** 16.09  
- **MAPE:** 7.64%

---
### Forecast vs Actual
<img width="1600" height="800" alt="forecast_plot" src="https://github.com/user-attachments/assets/f6dd5c67-f946-43a7-8c7c-c1f7c0711891" />

---
### Training & Validation Loss
<img width="1120" height="800" alt="training_curves" src="https://github.com/user-attachments/assets/e06d924a-3f81-4984-bed2-a8e8b5f508aa" />

---

## Project Structure
```
lstm-time-series-forecasting/
├─ README.md
├─ LICENSE
├─ requirements.txt
├─ data/
│  └─ generate_series.py
├─ src/
│  ├─ train_lstm.py
│  ├─ evaluate.py
│  └─ utils.py
└─ outputs/
   ├─ metrics.json
   ├─ forecast_plot.png
   ├─ training_curves.png
   ├─ best_lstm.pt
   ├─ scaler.pkl
   └─ (auto-created figures & reports)
```

## Setup
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
```

## Generate Data
```bash
python data/generate_series.py --start 2020-01-01 --end 2025-12-31 --seed 42 --out data/daily_series.csv
```

## Train Model
```bash
python src/train_lstm.py --input data/daily_series.csv --horizon 30 --lookback 60 --epochs 30 --batch-size 64 --outdir outputs --seed 42
```

## Evaluate
```bash
python src/evaluate.py --input data/daily_series.csv --model outputs/best_lstm.pt --lookback 60 --horizon 30 --outdir outputs
```

**Outputs**
- `outputs/metrics.json` – RMSE, MAE, MAPE  
- `outputs/training_curves.png` – training & validation curves  
- `outputs/forecast_plot.png` – forecast visualization  
- `outputs/best_lstm.pt` – trained PyTorch model  
- `outputs/scaler.pkl` – fitted scaler

import csv, os
import numpy as np
import pandas as pd
import torch
from train_mamba_aqi import build_time_series_samples, split_data_by_timeline, standardize, TimeSeriesMambaRegressorNoLoc, TimeSeriesMambaRegressor

DATA='dataset/2025.csv'
LOC='vinhlong_vinhlongcity'
OUT='outputs/ts_vinhlong'
CKPT=os.path.join(OUT,'best_mamba_aqi.pt')

df=pd.read_csv(DATA)
df=df[df['location_key'].astype(str)==LOC].copy()
print('rows', len(df))

x_seq, loc_ids, y, y_ts, num_locations, feature_cols = build_time_series_samples(df, 'aqi', 24, 1)
train, val, test = split_data_by_timeline(x_seq, loc_ids, y, y_ts)
train, val, test, y_mean, y_std = standardize(train, val, test)

# create dataset tensors
x_test = torch.from_numpy(test.x_seq).float()
loc_test = torch.from_numpy(test.loc_ids).long()
y_test = torch.from_numpy(test.y).float()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# choose model
model = TimeSeriesMambaRegressorNoLoc(num_features=x_test.shape[-1], d_model=64, n_layers=2).to(device)
model.load_state_dict(torch.load(CKPT, map_location=device))
model.eval()

preds = []
with torch.no_grad():
    for i in range(0, len(x_test), 512):
        xb = x_test[i:i+512].to(device)
        locb = loc_test[i:i+512].to(device)
        p = model(xb, locb).cpu().numpy()
        preds.append(p)

preds = np.concatenate(preds, axis=0)
# denormalize
preds = preds * y_std + y_mean
true = y_test.numpy() * y_std + y_mean
abs_err = np.abs(true - preds)

out_path = os.path.join(OUT, 'test_predictions.csv')
import pandas as pd
pd.DataFrame({'y_true': true, 'y_pred': preds, 'abs_error': abs_err}).to_csv(out_path, index=False)
print('wrote', out_path)

import sys
import os
import importlib.util
import pandas as pd

# Load AirQualityFormatter directly from file to avoid package import issues
sys.path.insert(0, os.path.abspath('Transformer_Timeseries'))
aq_path = os.path.join('Transformer_Timeseries', 'data_formatters', 'air_quality.py')
spec = importlib.util.spec_from_file_location('aqf', aq_path)
aqm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(aqm)
AirQualityFormatter = aqm.AirQualityFormatter

df = pd.read_csv('dataset/2025.csv')
print('Loaded rows:', len(df))
fmt = AirQualityFormatter()
train, val, test = fmt.split_data(df)
print('Train rows:', len(train))
print('Val rows:', len(val))
print('Test rows:', len(test))
print('Feature inputs:', fmt.feature_inputs)

import pandas as pd
import numpy as np
import sklearn.preprocessing
import data_formatters.base
import data_formatters.utils as utils

GenericDataFormatter = data_formatters.base.GenericDataFormatter
DataTypes = data_formatters.base.DataTypes
InputTypes = data_formatters.base.InputTypes

class AirQualityFormatter(GenericDataFormatter):

    _column_definition = [
        ('location_key', DataTypes.CATEGORICAL, InputTypes.ID),
        ('ts_utc', DataTypes.DATE, InputTypes.TIME),
        ('aqi', DataTypes.REAL_VALUED, InputTypes.TARGET),
        
        # Numeric observed inputs (aligned with Mamba numeric selection)
        ('pm25', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('pm10', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('no2', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('o3', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('so2', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('co', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('aod', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('dust', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('uv_index', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('co2', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        # Additional derived/auxiliary numeric features present in dataset
        ('aqi_pm25', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('aqi_pm10', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('aqi_no2', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('aqi_o3', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('aqi_so2', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('aqi_co', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        # Keep static categorical input at the end (matches TFT-style indexing)
        ('location_key', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),
    ]

    def __init__(self, selected_location=None, selected_locations=None):
        self.identifiers = None
        self._global_real_scaler = None
        self._global_target_scaler = None
        self._time_steps = 25 
        if selected_locations is None:
            if selected_location is None:
                self.selected_locations = []
            else:
                self.selected_locations = [str(selected_location)]
        elif isinstance(selected_locations, str):
            self.selected_locations = [x.strip() for x in selected_locations.split(',') if x and x.strip()]
        else:
            self.selected_locations = [str(x).strip() for x in selected_locations if str(x).strip()]
        self.selected_location = self.selected_locations[0] if len(self.selected_locations) == 1 else None
        self.feature_inputs = None

    def split_data(self, df):
        # Mamba-like: create windows first, then split timeline on windows
        print("Formatting train-valid-test splits using window-first global timeline (70/10/20)...")
        df = df.copy()

        column_definitions = self.get_column_definition()
        id_column = utils.get_single_col_by_input_type(InputTypes.ID, column_definitions)
        time_column = utils.get_single_col_by_input_type(InputTypes.TIME, column_definitions)
        target_column = utils.get_single_col_by_input_type(InputTypes.TARGET, column_definitions)

        # Ensure correct dtypes and drop rows with missing time/target
        df[time_column] = pd.to_datetime(df[time_column], utc=True, errors="coerce")
        df = df.dropna(subset=[time_column, id_column, target_column]).copy()
        df[target_column] = pd.to_numeric(df[target_column], errors="coerce")

        # Convert numeric columns and fill missing with median (like Mamba)
        real_inputs = utils.extract_cols_from_data_type(
            DataTypes.REAL_VALUED, column_definitions, set()
        )
        for col in real_inputs:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                fill_val = df[col].median()
                if pd.isna(fill_val):
                    fill_val = 0.0
                df[col] = df[col].fillna(fill_val)

        # Build window metadata per location: (end_ts, identifier, start, end)
        time_steps = int(self._time_steps)
        grouped = {}
        windows = []
        for identifier, g in df.sort_values([id_column, time_column]).groupby(id_column, sort=False):
            g = g.reset_index(drop=True)
            grouped[identifier] = g
            n = len(g)
            if n >= time_steps:
                for end in range(time_steps - 1, n):
                    start = end - time_steps + 1
                    windows.append((g.loc[end, time_column], identifier, start, end))

        if not windows:
            return (pd.DataFrame(columns=df.columns) for _ in range(3))

        windows.sort(key=lambda t: t[0])
        total_w = len(windows)
        n_train_w = int(total_w * 0.7)
        n_val_w = int(total_w * 0.1)

        train_w = windows[:n_train_w]
        val_w = windows[n_train_w:n_train_w + n_val_w]
        test_w = windows[n_train_w + n_val_w:]

        # Convert window assignments to row subsets for dataset preprocessing
        def rows_from_windows(win_meta):
            if not win_meta:
                return pd.DataFrame(columns=df.columns)

            masks = {
                identifier: np.zeros(len(g), dtype=bool)
                for identifier, g in grouped.items()
            }
            for _, identifier, start, end in win_meta:
                masks[identifier][start:end + 1] = True

            parts = []
            for identifier, g in grouped.items():
                m = masks[identifier]
                if m.any():
                    parts.append(g.loc[m])

            if not parts:
                return pd.DataFrame(columns=df.columns)

            out = pd.concat(parts, axis=0, ignore_index=True)
            return out.sort_values([id_column, time_column]).reset_index(drop=True)

        train = rows_from_windows(train_w)
        valid = rows_from_windows(val_w)
        test = rows_from_windows(test_w)

        # Calibrate scalers from train split only (Mamba-style train statistics)
        feature_inputs = [c for c in real_inputs if c != target_column]

        feat_matrix = train[feature_inputs].values.astype(float)
        target_matrix = train[[target_column]].values.astype(float)

        self._global_real_scaler = sklearn.preprocessing.StandardScaler().fit(feat_matrix)
        self._global_target_scaler = sklearn.preprocessing.StandardScaler().fit(target_matrix)
        self._global_real_scaler.scale_ = np.where(self._global_real_scaler.scale_ < 1e-6, 1.0, self._global_real_scaler.scale_)
        self._global_target_scaler.scale_ = np.where(self._global_target_scaler.scale_ < 1e-6, 1.0, self._global_target_scaler.scale_)
        self.identifiers = train[id_column].unique().tolist()
        self.feature_inputs = feature_inputs

        return (self.transform_inputs(x) for x in [train, valid, test])

    def set_scalers(self, df):
        column_definitions = self.get_column_definition()
        id_column = utils.get_single_col_by_input_type(InputTypes.ID, column_definitions)
        target_column = utils.get_single_col_by_input_type(InputTypes.TARGET, column_definitions)
        
        real_inputs = utils.extract_cols_from_data_type(
            DataTypes.REAL_VALUED, column_definitions, {InputTypes.ID, InputTypes.TIME})

        feature_inputs = [c for c in real_inputs if c != target_column]

        data = df[feature_inputs].values
        targets = df[[target_column]].values

        self._global_real_scaler = sklearn.preprocessing.StandardScaler().fit(data)
        self._global_target_scaler = sklearn.preprocessing.StandardScaler().fit(targets)
        
        self._global_real_scaler.scale_ = np.where(self._global_real_scaler.scale_ < 1e-6, 1.0, self._global_real_scaler.scale_)
        self._global_target_scaler.scale_ = np.where(self._global_target_scaler.scale_ < 1e-6, 1.0, self._global_target_scaler.scale_)

        self.identifiers = df[id_column].unique().tolist()
        self.feature_inputs = feature_inputs

    def transform_inputs(self, df):
        df = df.copy()
        column_definitions = self.get_column_definition()
        id_col = utils.get_single_col_by_input_type(InputTypes.ID, column_definitions)
        time_col = utils.get_single_col_by_input_type(InputTypes.TIME, column_definitions)
        target_col = utils.get_single_col_by_input_type(InputTypes.TARGET, column_definitions)

        df[time_col] = pd.to_datetime(df[time_col], utc=True).dt.tz_localize(None)

        df_list = []

        for identifier, sliced in df.groupby(id_col):
            if len(sliced) >= self._time_steps:
                sliced_copy = sliced.copy()
                sliced_copy[self.feature_inputs] = self._global_real_scaler.transform(sliced_copy[self.feature_inputs].values)
                if self._global_target_scaler is not None:
                    sliced_copy[[target_col]] = self._global_target_scaler.transform(sliced_copy[[target_col]].values)
                sliced_copy[id_col] = identifier
                df_list.append(sliced_copy)

        if not df_list:
             return pd.DataFrame(columns=df.columns)

        df = pd.concat(df_list, axis=0).reset_index(drop=True)
        df[time_col] = df[time_col].astype(str)
        return df

    def get_fixed_params(self):
        return {
            'total_time_steps': 25,
            'num_encoder_steps': 24,
            'num_epochs': 5,
            'early_stopping_patience': 5,
            'multiprocessing_workers': 5
        }

    def get_default_model_params(self):
        return {
            'dropout_rate': 0.1,
            'hidden_layer_size': 64,
            'learning_rate': 0.0003,
            'minibatch_size': 128,
            'num_heads': 4,
            'stack_size': 1
        }

    def format_predictions(self, predictions):
        if self._global_target_scaler is None:
            return predictions

        output = predictions.copy()
        output = self._global_target_scaler.inverse_transform(output)
        return output
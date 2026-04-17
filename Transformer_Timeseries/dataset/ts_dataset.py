from torch import from_numpy
import pandas as pd
import data_formatters.utils as utils
from data_formatters.base import InputTypes
from torch.utils.data import Dataset
import numpy as np
import click
from os import path
from sklearn.preprocessing import LabelEncoder

class TSDataset(Dataset):
    ## Mostly adapted from original TFT Github, data_formatters
    def __init__(self, cnf, data_formatter):

        self.params = cnf.all_params

        self.csv = utils.data_csv_path(cnf.ds_name)
        self.data = pd.read_csv(self.csv, index_col=0, na_filter=False)
        if 'id' not in self.data.columns:
            self.data.reset_index(inplace=True)
        # -----------------------------------------

        id_col = utils.get_single_col_by_input_type(InputTypes.ID, data_formatter.get_column_definition())
        
        # Chuyển đổi toàn bộ cột ID từ chữ sang số nguyên (0, 1, 2...)
        le = LabelEncoder()
        selected_locations = []
        if hasattr(data_formatter, 'selected_locations') and data_formatter.selected_locations is not None:
            raw = data_formatter.selected_locations
            if isinstance(raw, str):
                selected_locations = [x.strip() for x in raw.split(',') if x and x.strip()]
            else:
                selected_locations = [str(x).strip() for x in raw if str(x).strip()]
        elif hasattr(data_formatter, 'selected_location') and data_formatter.selected_location is not None:
            selected_locations = [str(data_formatter.selected_location).strip()]

        if selected_locations:
            self.data = self.data[self.data[id_col].astype(str).isin(selected_locations)].copy()
            if self.data.empty:
                raise ValueError(f"Không có dữ liệu cho các địa điểm: {selected_locations}")

        # Dòng code cũ của bạn (giữ nguyên):
        self.data[id_col] = le.fit_transform(self.data[id_col].astype(str))
        self.params['num_locations'] = int(self.data[id_col].max()) + 1
        self.selected_locations = selected_locations
        self.train_set, self.valid_set, self.test_set = data_formatter.split_data(self.data)
        self.params['column_definition'] = data_formatter.get_column_definition()

        self.inputs = None
        self.outputs = None
        self.time = None
        self.identifiers = None

    def train(self):
        max_samples = self.params['train_samples']
        cache_suffix = self._cache_suffix()
        cache_path = utils.csv_path_to_folder(self.csv) + f"processed_traindata{cache_suffix}.npz"
        if path.exists(cache_path):
            f = np.load(cache_path, allow_pickle=True)
            self.inputs, self.outputs, self.time, self.identifiers = f[f.files[0]], f[f.files[1]], f[f.files[2]], f[f.files[3]]
            expected_input = int(self.params['input_size'])
            expected_out = int(self.params['output_size'])
            if self.inputs.shape[-1] != expected_input or self.outputs.shape[-1] != expected_out:
                print('Cached train data shape mismatch with current config. Rebuilding cache...')
                self.preprocess(self.train_set, max_samples)
                np.savez(cache_path, self.inputs, self.outputs, self.time, self.identifiers)
        else:
            self.preprocess(self.train_set, max_samples)
            np.savez(cache_path, self.inputs, self.outputs, self.time, self.identifiers)

    def test(self):
        max_samples = self.params['test_samples']
        cache_suffix = self._cache_suffix()
        cache_path = utils.csv_path_to_folder(self.csv) + f"processed_testdata{cache_suffix}.npz"
        if path.exists(cache_path):
            f = np.load(cache_path, allow_pickle=True)
            self.inputs, self.outputs, self.time, self.identifiers = f[f.files[0]], f[f.files[1]], f[f.files[2]], f[f.files[3]]
            expected_input = int(self.params['input_size'])
            expected_out = int(self.params['output_size'])
            if self.inputs.shape[-1] != expected_input or self.outputs.shape[-1] != expected_out:
                print('Cached test data shape mismatch with current config. Rebuilding cache...')
                self.preprocess(self.test_set, max_samples)
                np.savez(cache_path, self.inputs, self.outputs, self.time, self.identifiers)
        else:
            self.preprocess(self.test_set, max_samples)
            np.savez(cache_path, self.inputs, self.outputs, self.time, self.identifiers)

    def val(self):
        max_samples = self.params['val_samples']
        cache_suffix = self._cache_suffix()
        cache_path = utils.csv_path_to_folder(self.csv) + f"processed_validdata{cache_suffix}.npz"
        if path.exists(cache_path):
            f = np.load(cache_path, allow_pickle=True)
            self.inputs, self.outputs, self.time, self.identifiers = f[f.files[0]], f[f.files[1]], f[f.files[2]], f[f.files[3]]
            expected_input = int(self.params['input_size'])
            expected_out = int(self.params['output_size'])
            if self.inputs.shape[-1] != expected_input or self.outputs.shape[-1] != expected_out:
                print('Cached val data shape mismatch with current config. Rebuilding cache...')
                self.preprocess(self.valid_set, max_samples)
                np.savez(cache_path, self.inputs, self.outputs, self.time, self.identifiers)
        else:
            self.preprocess(self.valid_set, max_samples)
            np.savez(cache_path, self.inputs, self.outputs, self.time, self.identifiers)

    def preprocess(self, data, max_samples):
        time_steps = int(self.params['total_time_steps'])
        input_size = int(self.params['input_size'])
        output_size = int(self.params['output_size'])
        column_definition = self.params['column_definition']

        id_col = self._get_single_col_by_type(InputTypes.ID)
        time_col = self._get_single_col_by_type(InputTypes.TIME)

        data.sort_values(by=[id_col, time_col], inplace=True)
        print('Getting valid sampling locations.')
        valid_sampling_locations = []
        split_data_map = {}
        for identifier, df in data.groupby(id_col):
            # print('Getting locations for {}'.format(identifier))
            num_entries = len(df)
            if num_entries >= time_steps:
                valid_sampling_locations += [
                    (identifier, time_steps + i)
                    for i in range(num_entries - time_steps + 1)
                ]
            split_data_map[identifier] = df

        alloc_samples = max_samples if max_samples is not None and int(max_samples) > 0 else len(valid_sampling_locations)
        self.inputs = np.zeros((alloc_samples, time_steps, input_size))
        self.outputs = np.zeros((alloc_samples, time_steps, output_size))
        self.time = np.empty((alloc_samples, time_steps, 1), dtype=object)
        self.identifiers = np.empty((alloc_samples, time_steps, 1), dtype=object)
        print('# available segments={}'.format(len(valid_sampling_locations)))

        if max_samples is not None and int(max_samples) > 0 and len(valid_sampling_locations) > int(max_samples):
            print('Extracting first {} segments in chronological order...'.format(int(max_samples)))
            ranges = valid_sampling_locations[:int(max_samples)]
        else:
            print('Max samples={} exceeds # available segments={}'.format(
                max_samples, len(valid_sampling_locations)))
            ranges = valid_sampling_locations
            max_samples = len(valid_sampling_locations)
            self.inputs = self.inputs[:max_samples]
            self.outputs = self.outputs[:max_samples]
            self.time = self.time[:max_samples]
            self.identifiers = self.identifiers[:max_samples]

        id_col = self._get_single_col_by_type(InputTypes.ID)
        time_col = self._get_single_col_by_type(InputTypes.TIME)
        target_col = self._get_single_col_by_type(InputTypes.TARGET)
        input_cols = [
            tup[0]
            for tup in column_definition
            if tup[2] not in {InputTypes.ID, InputTypes.TIME}
        ]

        for i, tup in enumerate(ranges):
            if ((i + 1) % 1000) == 0:
                print(i + 1, 'of', max_samples, 'samples done...')
            identifier, start_idx = tup
            sliced = split_data_map[identifier].iloc[start_idx - time_steps:start_idx]

            self.inputs[i, :, :] = sliced[input_cols]
            self.outputs[i, :, :] = sliced[[target_col]]
            self.time[i, :, 0] = sliced[time_col]
            self.identifiers[i, :, 0] = sliced[id_col]

    def __getitem__(self, index):

        num_encoder_steps = int(self.params['num_encoder_steps'])
        s = {
            'inputs': self.inputs[index].astype(float),
            'outputs': self.outputs[index, num_encoder_steps:, :],
            'active_entries': np.ones_like(self.outputs[index, num_encoder_steps:, :]),
#           'time': self.time[index].tolist(),
            'identifier': self.identifiers[index].tolist()
        }

        return s

    def __len__(self):
        return self.inputs.shape[0]

    def _get_single_col_by_type(self, input_type):
        """Returns name of single column for input type."""
        return utils.get_single_col_by_input_type(input_type, self.params['column_definition'])

    def _cache_suffix(self):
        if not self.selected_locations:
            return ""
        safe = [str(x).lower().replace(' ', '_').replace('-', '_') for x in self.selected_locations]
        safe = ["".join([ch for ch in s if ch.isalnum() or ch == '_']) for s in safe]
        return "_" + "_".join(safe)


@click.command()
@click.option('--conf_file_path', type=str, default="./conf/electricity.yaml")
def main(conf_file_path):
    import data_formatters.utils as utils
    from conf import Conf

    cnf = Conf(conf_file_path=conf_file_path, seed=15, exp_name="test", log=False)
    data_formatter = utils.make_data_formatter(cnf.ds_name)
    dataset_train = TSDataset(cnf, data_formatter)
    dataset_train.train()

    for i in range(10):
        # 192 x ['power_usage', 'hour', 'day_of_week', 'hours_from_start', 'categorical_id']
        x = dataset_train[i]['inputs']
        # 24 x ['power_usage']
        y = dataset_train[i]['outputs']
        print(f'Example #{i}: x.shape={x.shape}, y.shape={y.shape}')


if __name__ == "__main__":
    main()
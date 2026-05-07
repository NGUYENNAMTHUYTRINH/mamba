# -*- coding: utf-8 -*-
# ---------------------

from time import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from conf import Conf
from dataset.ts_dataset import TSDataset
from models.temporal_fusion_t import tft_model
from progress_bar import ProgressBar
from utils import QuantileLoss, symmetric_mean_absolute_percentage_error, unnormalize_tensor
import data_formatters.utils as utils
from models.transformer import Transformer
from models.transformer_grn.transformer import Transformer as GRNTransformer

class Trainer(object):
    """
    Class for training and test the model
    """

    def __init__(self, cnf):
        # type: (Conf) -> Trainer

        torch.set_num_threads(3)

        self.cnf = cnf

        # Read selected locations from CLI config (supports 1 or many).
        selected_locations = getattr(cnf, 'selected_locations', None)
        if selected_locations is None:
            loc = getattr(cnf, 'selected_location', None)
            selected_locations = [loc] if loc else []
        self.selected_locations = [str(x) for x in selected_locations if str(x).strip()]

        if cnf.ds_name == "air_quality":
            # Nếu chạy air_quality, import class và truyền biến loc vào
            from data_formatters.air_quality import AirQualityFormatter
            feature_cols = self.cnf.all_params.get("feature_cols")
            target_col = self.cnf.all_params.get("target_col", "aqi")
            self.data_formatter = AirQualityFormatter(
                selected_locations=self.selected_locations,
                feature_cols=feature_cols,
                target_col=target_col,
            )
        else:
            self.data_formatter = utils.make_data_formatter(cnf.ds_name)

        loader = TSDataset

        # init dataset
        dataset_train = loader(self.cnf, self.data_formatter)
        dataset_train.train()
        dataset_test = loader(self.cnf, self.data_formatter)
        dataset_test.test()

        # init model
        model_choice = self.cnf.all_params["model"]
        if model_choice == "transformer":
            self.model = Transformer(self.cnf.all_params)
        elif model_choice == "tf_transformer":
            self.model = tft_model.TFT(self.cnf.all_params)
        elif model_choice == "grn_transformer":
            self.model = GRNTransformer(self.cnf.all_params)
        else:
            raise NameError

        self.model = self.model.to(cnf.device)

        self.model_choice = model_choice
        self.point_mode = bool(self.cnf.all_params.get('point_forecast', True))
        self.use_quantile_loss = (self.model_choice == 'tf_transformer' and
                                  bool(self.cnf.all_params.get('use_quantile_loss_for_tft', True)))

        # init optimizer (Mamba-like defaults)
        optimizer_name = str(self.cnf.all_params.get('optimizer', 'adamw')).lower()
        weight_decay = float(self.cnf.all_params.get('weight_decay', 1e-4))
        if optimizer_name == 'adam':
            self.optimizer = optim.Adam(params=self.model.parameters(), lr=cnf.lr, weight_decay=weight_decay)
        else:
            self.optimizer = optim.AdamW(params=self.model.parameters(), lr=cnf.lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=float(self.cnf.all_params.get('lr_reduce_factor', 0.5)),
            patience=int(self.cnf.all_params.get('lr_patience', 2)),
            min_lr=float(self.cnf.all_params.get('min_lr', 1e-6)),
        )

        # point forecast objective (Mamba-like)
        loss_name = str(self.cnf.all_params.get('loss', 'huber')).lower()
        if loss_name == 'mse':
            self.loss_fn = nn.MSELoss()
        else:
            self.loss_fn = nn.HuberLoss(delta=float(self.cnf.all_params.get('huber_delta', 1.0)))
        self.quantile_loss = QuantileLoss(cnf.quantiles)

        self.use_amp = bool(self.cnf.all_params.get('amp', False)) and str(cnf.device).startswith('cuda')
        self.grad_accum_steps = max(1, int(self.cnf.all_params.get('grad_accum_steps', 1)))
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)

        # init train loader
        self.train_loader = DataLoader(
            dataset=dataset_train, batch_size=cnf.batch_size,
            num_workers=cnf.n_workers, shuffle=False, pin_memory=True,
        )

        # init test loader
        self.test_loader = DataLoader(
            dataset=dataset_test, batch_size=cnf.batch_size,
            num_workers=cnf.n_workers, shuffle=False, pin_memory=True,
        )

        # init logging stuffs
        self.log_path = cnf.exp_log_path
        print(f'tensorboard --logdir={str(cnf.project_log_path)}\n')
        self.sw = SummaryWriter(self.log_path)
        self.log_freq = len(self.train_loader)
        self.train_losses = []
        self.test_loss = []
        self.test_losses = {'p10': [], 'p50': [], 'p90': []}
        self.test_smape = []
        self.metrics_history_path = self.log_path / 'metrics_history.csv'
        self.metrics_history_rows = []
        if self.metrics_history_path.exists():
            try:
                self.metrics_history_rows = pd.read_csv(self.metrics_history_path).to_dict(orient='records')
            except Exception:
                self.metrics_history_rows = []

        # starting values
        self.epoch = 0
        self.best_test_loss = None
        self.early_stopping_patience = int(self.cnf.all_params.get('early_stopping_patience', 5))
        self.no_improve_epochs = 0

        # init progress bar
        self.progress_bar = ProgressBar(max_step=self.log_freq, max_epoch=self.cnf.epochs)

        # possibly load checkpoint
        self.load_ck()

        print("Finished preparing datasets.")

    def _extract_point_prediction(self, output: torch.Tensor) -> torch.Tensor:
        if output.ndim == 3:
            if output.shape[-1] == 1:
                return output[..., 0]
            # if quantile-like output remains, use median channel
            return output[..., 1]
        if output.ndim == 2:
            return output
        return output.squeeze(-1)

    def _inverse_target(self, arr: np.ndarray) -> np.ndarray:
        scaler = getattr(self.data_formatter, '_global_target_scaler', None)
        if scaler is None:
            return arr

        orig_shape = arr.shape
        arr_2d = arr.reshape(-1, 1)
        arr_inv = scaler.inverse_transform(arr_2d)
        return arr_inv.reshape(orig_shape)

    def _save_predictions_csv(self, targets: np.ndarray, preds: np.ndarray, loc_ids: np.ndarray):
        loc_name = "all" if not self.selected_locations else "_".join([str(x) for x in self.selected_locations])
        loc_name = "".join([ch if ch.isalnum() or ch in ['_', '-'] else '_' for ch in str(loc_name)])
        out_df = pd.DataFrame({
            "location": loc_ids if loc_ids is not None and len(loc_ids) == len(preds) else [loc_name] * len(preds),
            "actual_aqi": targets,
            "predicted_aqi": preds,
        })
        file_name = self.log_path / f"transformer_predictions_{loc_name}.csv"
        out_df.to_csv(file_name, index=False)
        print(f"Da luu ket qua du doan (best) ra file: {file_name}")

    def load_ck(self):
        """
        load training checkpoint
        """
        ck_path = self.log_path / 'training.ck'
        if ck_path.exists():
            ck = torch.load(ck_path)
            print(f'[loading checkpoint \'{ck_path}\']')
            self.epoch = ck['epoch']
            self.progress_bar.current_epoch = self.epoch
            self.model.load_state_dict(ck['model'])
            self.optimizer.load_state_dict(ck['optimizer'])
            self.best_test_loss = ck.get('best_test_loss', self.best_test_loss)

    def save_ck(self):
        """
        save training checkpoint
        """
        ck = {
            'epoch': self.epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_test_loss': self.best_test_loss
        }
        torch.save(ck, self.log_path / 'training.ck')

    def train(self):
        """
        train model for one epoch on the Training-Set.
        """
        start_time = time()
        self.model.train()

        times = []
        self.optimizer.zero_grad(set_to_none=True)
        for step, sample in enumerate(self.train_loader):
            t = time()

            # Feed input to the model
            x = sample['inputs'].float().to(self.cnf.device)
            y = sample['outputs'].squeeze(-1).float().to(self.cnf.device)

            _dev_type = str(self.cnf.device).split(':')[0]  # 'cuda' or 'cpu'
            with torch.autocast(device_type=_dev_type, dtype=torch.float16, enabled=self.use_amp):
                if self.cnf.all_params["model"] == "tf_transformer":
                    output, _, _ = self.model.forward(x)
                else:
                    output = self.model.forward(x)
                if self.use_quantile_loss:
                    loss, _ = self.quantile_loss(output, y)
                else:
                    y_pred = self._extract_point_prediction(output)
                    loss = self.loss_fn(y_pred, y)
                loss_for_backward = loss / self.grad_accum_steps

            if not torch.isfinite(loss):
                print(f"\nWarning: non-finite loss at step={step}, skip batch")
                self.optimizer.zero_grad(set_to_none=True)
                continue

            if self.use_amp:
                self.scaler.scale(loss_for_backward).backward()
            else:
                loss_for_backward.backward()

            if ((step + 1) % self.grad_accum_steps == 0) or ((step + 1) == len(self.train_loader)):
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cnf.all_params['max_gradient_norm'])
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

            self.train_losses.append(loss.item())

            # print an incredible progress bar
            times.append(time() - t)
            if self.cnf.log_each_step or (not self.cnf.log_each_step and self.progress_bar.progress == 1):
                try:
                    print(f'\r{self.progress_bar} '
                          f'│ Loss: {np.mean(self.train_losses):.6f} '
                          f'│ ↯: {1 / np.mean(times):5.2f} step/s', end='')
                except UnicodeEncodeError:
                    # Fallback for Windows cp1252 consoles
                    print(f"\rLoss: {np.mean(self.train_losses):.6f} | speed: {1 / np.mean(times):5.2f} step/s", end='')
            self.progress_bar.inc()

        # log average loss of this epoch
        mean_epoch_loss = np.mean(self.train_losses)
        self.sw.add_scalar(tag='train_loss', scalar_value=mean_epoch_loss, global_step=self.epoch)
        self.train_losses = []

        # log epoch duration
        epoch_sec = time() - start_time
        print(f' | T: {epoch_sec:.2f} s')
        return {
            'train_loss': float(mean_epoch_loss),
            'epoch_sec': float(epoch_sec),
        }

    def test(self):
        """
        test model on the Test-Set
        """
        self.model.eval()
        output, sample = None, None
        self.test_loss = []
        self.test_smape = []
        self.test_losses = {'p10': [], 'p50': [], 'p90': []}

        all_targets = []
        all_preds = []
        all_loc_ids = []
        all_targets_norm = []
        all_preds_norm = []
        mae = np.nan
        rmse = np.nan
        mae_raw = np.nan
        rmse_raw = np.nan
        r2 = np.nan

        t = time()
        for step, sample in enumerate(self.test_loader):

            # Hide future predictions from input vector.
            # Set to 0 (normalized mean) for timesteps > encoder_steps, NOT 1.
            steps = self.cnf.all_params['num_encoder_steps']
            pred_len = sample['outputs'].shape[1]
            x = sample['inputs'].float().to(self.cnf.device)
            x[:, steps:, 0] = 0

            # Feed input to the model
            if self.cnf.all_params["model"] == "transformer" or self.cnf.all_params["model"] == "grn_transformer":

                # Auto-regressive prediction
                for i in range(pred_len):
                    output = self.model.forward(x)
                    point_step = self._extract_point_prediction(output)
                    x[:, steps + i, 0] = point_step[:, i]
                output = self.model.forward(x)

            elif self.cnf.all_params["model"] == "tf_transformer":
                output, _, _ = self.model.forward(x)
            else:
                raise NameError

            y = sample['outputs'].squeeze(-1).float().to(self.cnf.device)
            y_pred = self._extract_point_prediction(output)

            # Compute point loss
            if self.use_quantile_loss:
                loss, _ = self.quantile_loss(output, y)
            else:
                loss = self.loss_fn(y_pred, y)
            smape = symmetric_mean_absolute_percentage_error(
                y_pred.detach().cpu().numpy(),
                sample['outputs'][:, :, 0].detach().cpu().numpy()
            )

            # Inverse target scaling for metrics on original AQI scale
            y_np = y.detach().cpu().numpy()
            y_pred_np = y_pred.detach().cpu().numpy()
            target = self._inverse_target(y_np)
            pred_forecast = self._inverse_target(y_pred_np)

            # Collect unnormalized targets and predictions for Regression Metrics
            all_targets.append(target.flatten())
            all_preds.append(pred_forecast.flatten())
            all_targets_norm.append(y_np.flatten())
            all_preds_norm.append(y_pred_np.flatten())
            identifiers = sample.get('identifier', None)
            if identifiers is not None:
                id_arr = np.array(identifiers, dtype=object)
                if id_arr.ndim >= 2:
                    # sample['identifier'] shape is typically (B, pred_len, 1)
                    id_arr = id_arr[:, -1]
                id_arr = id_arr.reshape(-1)
                all_loc_ids.append(id_arr.astype(str))

            if self.use_quantile_loss and output.ndim == 3 and output.shape[-1] >= 3:
                p10_forecast = self._inverse_target(output[..., 0].detach().cpu().numpy())
                p50_forecast = self._inverse_target(output[..., 1].detach().cpu().numpy())
                p90_forecast = self._inverse_target(output[..., 2].detach().cpu().numpy())
                self.test_losses['p10'].append(self.quantile_loss.numpy_normalised_quantile_loss(p10_forecast, target, 0.1))
                self.test_losses['p50'].append(self.quantile_loss.numpy_normalised_quantile_loss(p50_forecast, target, 0.5))
                self.test_losses['p90'].append(self.quantile_loss.numpy_normalised_quantile_loss(p90_forecast, target, 0.9))

            self.test_loss.append(loss.item())
            self.test_smape.append(smape)

        # log log log
        mean_test_loss = np.mean(self.test_loss)
        mean_smape = np.mean(self.test_smape)
        if self.use_quantile_loss:
            for k in self.test_losses.keys():
                if len(self.test_losses[k]) > 0:
                    mean_q = np.mean(self.test_losses[k])
                    print(f'\tAVG {k} Loss on TEST-set: {mean_q:.6f} | T: {time() - t:.2f} s')
                    self.sw.add_scalar(tag=k + '_test_loss', scalar_value=mean_q, global_step=self.epoch)
        print(f'\tAVG Loss on TEST-set: {mean_test_loss:.6f} | T: {time() - t:.2f} s')
        print(f'\tAVG SMAPE on TEST-set: {mean_smape:.6f} | T: {time() - t:.2f} s')
        self.sw.add_scalar(tag='test_smape', scalar_value=mean_smape, global_step=self.epoch)
        self.sw.add_scalar(tag='test_loss', scalar_value=mean_test_loss, global_step=self.epoch)

        # --- BẮT ĐẦU PHẦN TÍNH TOÁN METRICS GIỐNG MAMBA ---
        try:
            # Flatten tất cả list ra thành 1 mảng 1D
            final_targets = np.concatenate(all_targets)
            final_preds = np.concatenate(all_preds)

            mae_raw = mean_absolute_error(final_targets, final_preds)
            mse_raw = mean_squared_error(final_targets, final_preds)
            rmse_raw = np.sqrt(mse_raw)
            r2 = r2_score(final_targets, final_preds)

            final_targets_norm = np.concatenate(all_targets_norm)
            final_preds_norm = np.concatenate(all_preds_norm)
            mae = mean_absolute_error(final_targets_norm, final_preds_norm)
            mse = mean_squared_error(final_targets_norm, final_preds_norm)
            rmse = np.sqrt(mse)

            print("\n" + "="*50)
            print("KET QUA DANH GIA TREN TAP TEST (NORMALIZED)")
            print("="*50)
            print(f"MAE:  {mae:.4f}")
            print(f"RMSE: {rmse:.4f}")
            print(f"R2:   {r2:.4f}")
            print("="*50 + "\n")

        except Exception as e:
            print(f"Lỗi khi tính toán metrics so sánh: {e}")
        # --- KẾT THÚC PHẦN METRICS ---

        return {
            'test_loss': float(mean_test_loss),
            'test_smape': float(mean_smape),
            'test_mae': float(mae),
            'test_rmse': float(rmse),
            'test_mae_norm': float(mae),
            'test_rmse_norm': float(rmse),
            'test_mae_raw': float(mae_raw),
            'test_rmse_raw': float(rmse_raw),
            'test_r2': float(r2),
            'all_targets': np.concatenate(all_targets) if len(all_targets) > 0 else np.array([]),
            'all_preds': np.concatenate(all_preds) if len(all_preds) > 0 else np.array([]),
            'all_loc_ids': np.concatenate(all_loc_ids) if len(all_loc_ids) > 0 else np.array([]),
        }

    def run(self):
        """
        start model training procedure (train > test > checkpoint > repeat)
        """
        for _ in range(self.epoch, self.cnf.epochs):
            train_metrics = self.train()

            with torch.no_grad():
                test_metrics = self.test()

            # Save best checkpoint + best predictions for fair compare
            if self.best_test_loss is None or test_metrics['test_loss'] < self.best_test_loss:
                self.best_test_loss = test_metrics['test_loss']
                self.no_improve_epochs = 0
                torch.save(self.model.state_dict(), self.log_path / (self.cnf.exp_name + '_best.pth'))
                if test_metrics.get('all_targets') is not None and test_metrics.get('all_preds') is not None:
                    if len(test_metrics['all_targets']) > 0 and len(test_metrics['all_preds']) > 0:
                        self._save_predictions_csv(
                            test_metrics['all_targets'],
                            test_metrics['all_preds'],
                            test_metrics.get('all_loc_ids', np.array([])),
                        )
            else:
                self.no_improve_epochs += 1

            self.scheduler.step(test_metrics['test_loss'])
            current_lr = float(self.optimizer.param_groups[0]['lr'])

            self.metrics_history_rows.append({
                'epoch': int(self.epoch + 1),
                'train_loss': float(train_metrics.get('train_loss', np.nan)),
                'test_loss': float(test_metrics.get('test_loss', np.nan)),
                'test_smape': float(test_metrics.get('test_smape', np.nan)),
                'test_mae': float(test_metrics.get('test_mae', np.nan)),
                'test_rmse': float(test_metrics.get('test_rmse', np.nan)),
                'test_mae_norm': float(test_metrics.get('test_mae_norm', np.nan)),
                'test_rmse_norm': float(test_metrics.get('test_rmse_norm', np.nan)),
                'test_mae_raw': float(test_metrics.get('test_mae_raw', np.nan)),
                'test_rmse_raw': float(test_metrics.get('test_rmse_raw', np.nan)),
                'test_r2': float(test_metrics.get('test_r2', np.nan)),
                # Use 'train_sec' key (consistent with run_tft_pipeline lookup)
                'train_sec': float(train_metrics.get('epoch_sec', np.nan)),
                'epoch_sec': float(train_metrics.get('epoch_sec', np.nan)),
                'lr': current_lr,
            })
            pd.DataFrame(self.metrics_history_rows).to_csv(self.metrics_history_path, index=False)

            self.epoch += 1
            self.save_ck()

            if self.no_improve_epochs >= self.early_stopping_patience:
                print(f"Early stopping: no improvement for {self.no_improve_epochs} epochs.")
                break
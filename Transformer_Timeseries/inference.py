# -*- coding: utf-8 -*-
# ---------------------

from time import time
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from conf import Conf
from dataset.ts_dataset import TSDataset
from models.temporal_fusion_t import tft_model
from progress_bar import ProgressBar
from utils import QuantileLoss, symmetric_mean_absolute_percentage_error, unnormalize_tensor, plot_temporal_serie
import data_formatters.utils as utils
from models.transformer import Transformer
from models.transformer_grn.transformer import Transformer as GRNTransformer



class TS(object):
    """
    Class for loading and test the pre-trained model
    """

    def __init__(self, cnf):
        # type: (Conf) -> None

        self.cnf = cnf
        if cnf.ds_name == "air_quality":
            from data_formatters.air_quality import AirQualityFormatter
            feature_cols = self.cnf.all_params.get("feature_cols")
            target_col = self.cnf.all_params.get("target_col", "aqi")
            self.data_formatter = AirQualityFormatter(
                feature_cols=feature_cols,
                target_col=target_col,
            )
        else:
            self.data_formatter = utils.make_data_formatter(cnf.ds_name)

        loader = TSDataset
        dataset_test = loader(self.cnf, self.data_formatter)
        dataset_test.test()

        # init model
        model_choice = self.cnf.all_params["model"]
        if model_choice == "transformer":
            # Baseline transformer
            self.model = Transformer(self.cnf.all_params)
        elif model_choice == "tf_transformer":
            # Temporal fusion transformer
            self.model = tft_model.TFT(self.cnf.all_params)
        elif model_choice == "grn_transformer":
            # Transformer + GRN to encode static vars
            self.model = GRNTransformer(self.cnf.all_params)
        else:
            raise NameError

        self.model = self.model.to(cnf.device)
        self.model_choice = model_choice
        self.use_quantile_loss = (self.model_choice == 'tf_transformer' and
                                  bool(self.cnf.all_params.get('use_quantile_loss_for_tft', True)))

        # init optimizer
        self.optimizer = optim.AdamW(params=self.model.parameters(), lr=cnf.lr,
                                     weight_decay=float(self.cnf.all_params.get('weight_decay', 1e-4)))
        loss_name = str(self.cnf.all_params.get('loss', 'huber')).lower()
        if loss_name == 'mse':
            self.loss_fn = nn.MSELoss()
        else:
            self.loss_fn = nn.HuberLoss(delta=float(self.cnf.all_params.get('huber_delta', 1.0)))
        self.quantile_loss = QuantileLoss(cnf.quantiles)

        # init test loader
        self.test_loader = DataLoader(
            dataset=dataset_test, batch_size=cnf.batch_size,
            num_workers=cnf.n_workers, shuffle=False, pin_memory=True,
        )

        # init logging stuffs
        self.log_path = cnf.exp_log_path
        self.log_freq = len(self.test_loader)
        self.train_losses = []
        self.test_loss = []
        self.test_losses = {'p10': [], 'p50': [], 'p90': []}
        self.test_smape = []

        # starting values
        self.epoch = 0
        self.best_test_loss = None

        # init progress bar
        self.progress_bar = ProgressBar(max_step=self.log_freq, max_epoch=self.cnf.epochs)

        # possibly load checkpoint
        self.load_ck()

        print("Finished preparing datasets.")

    def _extract_point_prediction(self, output: torch.Tensor) -> torch.Tensor:
        if output.ndim == 3:
            if output.shape[-1] == 1:
                return output[..., 0]
            return output[..., 1]
        if output.ndim == 2:
            return output
        return output.squeeze(-1)

    def load_ck(self):
        """
        load training checkpoint
        """
        ck_path = self.log_path / self.cnf.exp_name + '_best.pth'
        if ck_path.exists():
            ck = torch.load(ck_path)
            print(f'[loading checkpoint \'{ck_path}\']')
            self.model.load_state_dict(ck)

    def test(self):
        """
        Quick test and plot prediction without saving or logging stuff on tensorboarc
        """
        with torch.no_grad():
            self.model.eval()
            pred_forecast, target = None, None

            t = time()
            for step, sample in enumerate(self.test_loader):

                # Hide future predictions from input vector, set to 0 (or 1) values where timestep > encoder_steps
                steps = self.cnf.all_params['num_encoder_steps']
                pred_len = sample['outputs'].shape[1]
                x = sample['inputs'].float().to(self.cnf.device)
                x[:, steps:, 0] = 1

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

                # Compute loss
                if self.use_quantile_loss:
                    loss, _ = self.quantile_loss(output, y)
                else:
                    loss = self.loss_fn(y_pred, y)
                smape = symmetric_mean_absolute_percentage_error(y_pred.detach().cpu().numpy(),
                                                                 sample['outputs'][:, :, 0].detach().cpu().numpy())

                # De-Normalize to compute metrics
                target = unnormalize_tensor(self.data_formatter, y, sample['identifier'][0][0])
                pred_forecast = unnormalize_tensor(self.data_formatter, y_pred, sample['identifier'][0][0])

                if self.use_quantile_loss and output.ndim == 3 and output.shape[-1] >= 3:
                    p10_forecast = unnormalize_tensor(self.data_formatter, output[..., 0], sample['identifier'][0][0])
                    p50_forecast = unnormalize_tensor(self.data_formatter, output[..., 1], sample['identifier'][0][0])
                    p90_forecast = unnormalize_tensor(self.data_formatter, output[..., 2], sample['identifier'][0][0])
                    self.test_losses['p10'].append(self.quantile_loss.numpy_normalised_quantile_loss(p10_forecast, target, 0.1))
                    self.test_losses['p50'].append(self.quantile_loss.numpy_normalised_quantile_loss(p50_forecast, target, 0.5))
                    self.test_losses['p90'].append(self.quantile_loss.numpy_normalised_quantile_loss(p90_forecast, target, 0.9))

                self.test_loss.append(loss.item())
                self.test_smape.append(smape)

            # Plot serie prediction
            p = np.expand_dims(pred_forecast, axis=-1)
            target = np.expand_dims(target, axis=-1)
            plot_temporal_serie(p, target)

            if self.use_quantile_loss:
                for k in self.test_losses.keys():
                    if len(self.test_losses[k]) > 0:
                        mean_q = np.mean(self.test_losses[k])
                        print(f'\t● AVG {k} Loss on TEST-set: {mean_q:.6f} │ T: {time() - t:.2f} s')

            # log log log
            mean_test_loss = np.mean(self.test_loss)
            mean_smape = np.mean(self.test_smape)
            print(f'\t● AVG Loss on TEST-set: {mean_test_loss:.6f} │ T: {time() - t:.2f} s')
            print(f'\t● AVG SMAPE on TEST-set: {mean_smape:.6f} │ T: {time() - t:.2f} s')

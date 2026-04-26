"""
mamba/mamba_model.py
--------------------
Kiến trúc mạng Mamba cho bài toán time-series AQI forecasting.

Classes:
- TimeSeriesMambaRegressor      : model chính, có location embedding
- TimeSeriesMambaRegressorNoLoc : variant không dùng location embedding
                                  (dùng khi chỉ train 1 location)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from mamba_ssm import Mamba


# ---------------------------------------------------------------------------
# Model chính — có location embedding
# ---------------------------------------------------------------------------

class TimeSeriesMambaRegressor(nn.Module):
    """Mamba-based time-series regressor với location embedding.

    Flow:
        x_seq (B, T, F)
        → feature_proj (B, T, d_model)
        → cat với loc_token (B, 1, d_model) → (B, T+1, d_model)
        → n_layers Mamba blocks
        → LayerNorm
        → lấy last token (B, d_model)
        → head MLP → scalar (B,)

    Parameters
    ----------
    num_features  : số lượng feature đầu vào (F)
    num_locations : số lượng location (dùng cho Embedding)
    d_model       : chiều ẩn của Mamba
    n_layers      : số Mamba block xếp chồng
    """

    def __init__(
        self,
        num_features:  int,
        num_locations: int,
        d_model:  int = 64,
        n_layers: int = 2,
    ) -> None:
        super().__init__()
        self.feature_proj  = nn.Linear(num_features, d_model)
        self.location_emb  = nn.Embedding(num_locations, d_model)
        self.layers = nn.ModuleList([
            Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2, use_fast_path=False)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.num_features = num_features

    def forward(self, x_seq: torch.Tensor, loc_ids: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x_seq   : (B, T, F)
        loc_ids : (B,) — index của location

        Returns
        -------
        (B,) — giá trị dự đoán cho từng sample trong batch
        """
        x         = self.feature_proj(x_seq)                  # (B, T, d_model)
        loc_token = self.location_emb(loc_ids).unsqueeze(1)   # (B, 1, d_model)
        x         = torch.cat([loc_token, x], dim=1)          # (B, T+1, d_model)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        return self.head(x[:, -1, :]).squeeze(-1)              # lấy last token


# ---------------------------------------------------------------------------
# Variant — không có location embedding
# ---------------------------------------------------------------------------

class TimeSeriesMambaRegressorNoLoc(nn.Module):
    """Variant KHÔNG dùng location embedding.

    Dùng khi chỉ train trên 1 location duy nhất.
    Giữ nguyên tham số `loc_ids` trong forward để tương thích
    với DataLoader/AQIDataset hiện tại (chỉ bỏ qua, không dùng).

    Parameters
    ----------
    num_features : số lượng feature đầu vào (F)
    d_model      : chiều ẩn của Mamba
    n_layers     : số Mamba block xếp chồng
    """

    def __init__(
        self,
        num_features: int,
        d_model:  int = 64,
        n_layers: int = 2,
    ) -> None:
        super().__init__()
        self.feature_proj = nn.Linear(num_features, d_model)
        self.layers = nn.ModuleList([
            Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2, use_fast_path=False)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, x_seq: torch.Tensor, loc_ids: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x_seq   : (B, T, F)
        loc_ids : (B,) — không dùng, chỉ giữ để tương thích với AQIDataset

        Returns
        -------
        (B,) — giá trị dự đoán
        """
        x = self.feature_proj(x_seq)   # (B, T, d_model)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.head(x[:, -1, :]).squeeze(-1)
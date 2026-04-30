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
        → concat location embedding per-timestep → (B, T, F + E)
        → input_proj → (B, T, d_model)
        → n_layers Mamba blocks
        → LayerNorm
        → lấy last timestep (B, d_model)
        → head MLP → scalar (B,)

    Parameters
    ----------
    num_features   : số lượng feature đầu vào (F)
    num_locations  : số lượng location (dùng cho Embedding)
    d_model        : chiều ẩn của Mamba
    n_layers       : số Mamba block xếp chồng
    loc_embed_dim  : chiều embedding cho location
    """

    def __init__(
        self,
        num_features:  int,
        num_locations: int,
        d_model:  int = 64,
        n_layers: int = 2,
        loc_embed_dim: int = 8,
    ) -> None:
        super().__init__()
        self.location_emb = nn.Embedding(num_locations, loc_embed_dim)
        self.input_proj = nn.Linear(num_features + loc_embed_dim, d_model)
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
        loc_vec = self.location_emb(loc_ids)  # (B, E)
        loc_seq = loc_vec.unsqueeze(1).expand(-1, x_seq.size(1), -1)  # (B, T, E)
        x = torch.cat([x_seq, loc_seq], dim=-1)  # (B, T, F+E)
        x = self.input_proj(x)  # (B, T, d_model)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        return self.head(x[:, -1, :]).squeeze(-1)              # lấy last timestep


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
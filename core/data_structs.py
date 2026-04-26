"""
core/data_structs.py
--------------------
Các cấu trúc dữ liệu dùng chung cho mọi model (Mamba, LSTM, TFT).

Gồm:
- SplitData  : dataclass chứa x_seq / loc_ids / y sau khi split
- AQIDataset : PyTorch Dataset bọc SplitData thành tensor
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class SplitData:
    """Chứa một tập dữ liệu (train / val / test) dưới dạng numpy array.

    Attributes
    ----------
    x_seq   : (N, T, F) — chuỗi feature đầu vào
    loc_ids : (N,)      — id số nguyên của từng location
    y       : (N,)      — giá trị target cần dự đoán
    """
    x_seq:   np.ndarray
    loc_ids: np.ndarray
    y:       np.ndarray


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class AQIDataset(Dataset):
    """PyTorch Dataset bọc SplitData, tự động convert sang tensor.

    Trả về tuple (x_seq, loc_ids, y) trong mỗi __getitem__.
    """

    def __init__(self, split: SplitData) -> None:
        self.x_seq   = torch.from_numpy(split.x_seq).float()
        self.loc_ids = torch.from_numpy(split.loc_ids).long()
        self.y       = torch.from_numpy(split.y).float()

    def __len__(self) -> int:
        return self.y.shape[0]

    def __getitem__(self, idx):
        return self.x_seq[idx], self.loc_ids[idx], self.y[idx]
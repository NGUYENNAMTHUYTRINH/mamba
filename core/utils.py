"""
core/utils.py
-------------
Các hàm tiện ích dùng chung: logger, seed, device.

Hàm:
- setup_logger(out_dir, name)  → logging.Logger
- resolve_device(device_arg)   → torch.device
- set_seed(seed)               → None
"""

from __future__ import annotations

import logging
import os

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

def setup_logger(out_dir: str, name: str = "trainer") -> logging.Logger:
    """Tạo logger ghi ra console lẫn file out_dir/train.log.

    Parameters
    ----------
    out_dir : thư mục output — file train.log sẽ được tạo ở đây.
    name    : tên của logger (dùng để phân biệt khi có nhiều logger).

    Returns
    -------
    logging.Logger đã cấu hình sẵn handler.
    """
    os.makedirs(out_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    # Xóa handler cũ để tránh duplicate khi gọi lại hàm
    logger.handlers = []

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    # Console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File
    log_path = os.path.join(out_dir, "train.log")
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def resolve_device(device_arg: str) -> torch.device:
    """Chuyển chuỗi device_arg thành torch.device.

    Parameters
    ----------
    device_arg : "auto" | "cuda" | "cpu"
        - "auto" : tự chọn cuda nếu có, ngược lại dùng cpu.
        - "cuda" : bắt buộc dùng GPU, raise lỗi nếu không có.
        - "cpu"  : luôn dùng CPU.

    Returns
    -------
    torch.device
    """
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "Yêu cầu --device cuda nhưng CUDA không khả dụng. "
            "Kiểm tra lại GPU / PyTorch CUDA setup."
        )
    return torch.device(device_arg)


# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Thiết lập seed cho Python, NumPy và PyTorch để đảm bảo reproducibility.

    Parameters
    ----------
    seed : giá trị seed nguyên dương.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
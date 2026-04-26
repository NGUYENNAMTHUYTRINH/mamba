"""
utils.py
--------
Các hàm tiện ích dùng chung cho toàn bộ app (normalize, format, v.v.)
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Normalize / format helpers
# ---------------------------------------------------------------------------

def normalize_locations(value) -> list[str]:
    """Chuẩn hoá đầu vào thành list[str] cho selected_locations."""
    if value is None:
        return []
    if isinstance(value, str):
        return [x.strip() for x in value.split(",") if x and x.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    s = str(value).strip()
    return [s] if s else []


def format_time_utc_strings(values: pd.Series) -> pd.Series:
    """Format datetimes as YYYY-MM-DD HH:MM:SS+00:00 in UTC."""
    ts = pd.to_datetime(values, utc=True, errors="coerce")
    return ts.dt.strftime("%Y-%m-%d %H:%M:%S+00:00")


def synthesize_tft_time_from_dataset(repo_root: Path, locations: pd.Series) -> pd.Series:
    """Tạo hourly UTC timestamps per location khi TFT output không có time column."""
    dataset_path = repo_root / "dataset" / "2025.csv"
    loc_series = locations.astype(str).reset_index(drop=True)
    out = pd.Series(index=loc_series.index, dtype="object")

    base_map: dict[str, pd.Timestamp] = {}
    global_base = pd.Timestamp("2025-01-01 00:00:00", tz="UTC")

    if dataset_path.exists():
        try:
            src = pd.read_csv(dataset_path, usecols=["location_key", "ts_utc"])
            src["ts_utc"] = pd.to_datetime(src["ts_utc"], utc=True, errors="coerce")
            src = src.dropna(subset=["location_key", "ts_utc"]).copy()
            if not src.empty:
                max_per_loc = src.groupby(src["location_key"].astype(str))["ts_utc"].max()
                for k, v in max_per_loc.items():
                    base_map[str(k)] = v + pd.Timedelta(hours=1)
                global_base = src["ts_utc"].max() + pd.Timedelta(hours=1)
        except Exception:
            pass

    for loc, idx in loc_series.groupby(loc_series).groups.items():
        idx_list = list(idx)
        start = base_map.get(str(loc), global_base)
        rng = pd.date_range(start=start, periods=len(idx_list), freq="h", tz="UTC")
        out.loc[idx_list] = rng.strftime("%Y-%m-%d %H:%M:%S+00:00")

    return out


# ---------------------------------------------------------------------------
# Dynamic module loader
# ---------------------------------------------------------------------------

def load_train_module():
    """Load động mamba/train_mamba_aqi.py và trả về module.

    Sau khi tách cấu trúc, file nằm ở:
        project_root/mamba/train_mamba_aqi.py   (không còn thư mục scripts/)

    Cần thêm project_root vào sys.path để train_mamba_aqi.py tìm được:
        from core.data_structs import ...
        from core.metrics import ...
        from core.utils import ...
        from mamba.mamba_model import ...
    """
    import sys

    try:
        project_root = Path(__file__).parent.parent          # app/ -> project root
        mod_path = project_root / "mamba" / "train_mamba_aqi.py"

        if not mod_path.exists():
            raise FileNotFoundError(f"Không tìm thấy: {mod_path}")

        # Thêm project_root vào sys.path để các import "from core.xxx" trong
        # train_mamba_aqi.py hoạt động đúng khi load động bằng importlib.
        root_str = str(project_root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)

        spec = importlib.util.spec_from_file_location(
            "train_mamba_aqi_for_streamlit", str(mod_path)
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    except Exception:
        return None


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def build_future_24h_frame(
    df_valid: pd.DataFrame, feature_cols: list[str], target_col: str
) -> pd.DataFrame:
    """Tạo DataFrame dự báo 24h tiếp theo từ ngày cuối trong df_valid."""
    if "ts_utc" not in df_valid.columns:
        raise ValueError("Cần có cột 'ts_utc' để dự báo 24h tiếp theo.")

    work = df_valid.copy()
    work["ts_utc"] = pd.to_datetime(work["ts_utc"], utc=True, errors="coerce")
    work = work.dropna(subset=["ts_utc"]).copy()

    future_rows = []
    if "location_key" in work.columns:
        groups = [
            (loc, work.loc[work["location_key"].astype(str) == loc].sort_values("ts_utc").copy())
            for loc in sorted(work["location_key"].dropna().astype(str).unique().tolist())
        ]
    else:
        groups = [(None, work.sort_values("ts_utc").copy())]

    for loc, g in groups:
        if g.empty:
            continue

        last_ts = g["ts_utc"].iloc[-1]
        next_day_start = last_ts.normalize() + pd.Timedelta(days=1)
        template = g.tail(24).copy()
        if len(template) < 24:
            template = pd.concat(
                [template] * (24 // len(template) + 1), ignore_index=True
            ).head(24)

        for h in range(24):
            src = template.iloc[h].copy()
            row = {col: src[col] for col in feature_cols if col in template.columns}
            if loc is not None:
                row["location_key"] = loc
            row["ts_utc"] = next_day_start + pd.Timedelta(hours=h)
            row[target_col] = np.nan
            future_rows.append(row)

    if not future_rows:
        raise ValueError("Không tạo được dữ liệu dự báo 24h.")

    return pd.DataFrame(future_rows)


def split_data_by_timeline(
    x_seq: np.ndarray,
    loc_ids: np.ndarray,
    y: np.ndarray,
    y_ts: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
):
    """Fallback split theo timeline khi module scripts không load được."""
    if len(y) < 3:
        raise ValueError("Need at least 3 samples for train/val/test split.")

    order = np.argsort(y_ts)
    n = len(order)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    if train_end <= 0 or val_end <= train_end or val_end >= n:
        raise ValueError("Invalid timeline split sizes.")

    class _Split:
        def __init__(self, x, l, yy):
            self.x_seq = x
            self.loc_ids = l
            self.y = yy

    train_idx = order[:train_end]
    val_idx = order[train_end:val_end]
    test_idx = order[val_end:]
    return (
        _Split(x_seq[train_idx], loc_ids[train_idx], y[train_idx]),
        _Split(x_seq[val_idx], loc_ids[val_idx], y[val_idx]),
        _Split(x_seq[test_idx], loc_ids[test_idx], y[test_idx]),
    )
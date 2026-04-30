"""
pipeline_tft.py
---------------
Pipeline chạy Transformer (TFT) bằng subprocess và thu thập kết quả.
"""

from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from Utils import format_time_utc_strings, normalize_locations, synthesize_tft_time_from_dataset


def run_tft_pipeline(
    selected_locations: list[str],
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    loss_name: str,
    seed: int,
    use_gpu: bool,
    run_dir: str,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Chạy Transformer_Timeseries TFT pipeline, trả về (summary, history_df, pred_df).

    Dùng subprocess để tránh xung đột import giữa root repo và Transformer_Timeseries.
    """
    # app/ nằm trong project_root/app/, Transformer_Timeseries nằm ngang hàng
    repo_root = Path(__file__).parent.parent
    tft_root = repo_root / "Transformer_Timeseries"
    base_conf_path = tft_root / "conf" / "air_quality.yaml"

    if not base_conf_path.exists():
        raise FileNotFoundError(f"Không tìm thấy config TFT: {base_conf_path}")

    with open(base_conf_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["model"] = "tf_transformer"
    cfg["num_epochs"] = int(epochs)
    cfg["batch_size"] = int(batch_size)
    cfg["n_batch_size"] = int(batch_size)
    cfg["lr"] = float(lr)
    cfg["weight_decay"] = float(weight_decay)
    cfg["loss"] = str(loss_name)
    cfg["device"] = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"
    cfg["seed"] = int(seed)
    cfg["point_forecast"] = True
    cfg["use_quantile_loss_for_tft"] = False
    cfg["quantiles"] = [0.5]

    # ── Xác định đường dẫn CSV đầu vào ──────────────────────────────────
    # ưu tiên: session_state["data_path"] (có thể từ path nhập tay hoặc file upload tạm)
    import streamlit as _st
    _ss_path = _st.session_state.get("data_path", None)
    if _ss_path and Path(str(_ss_path)).exists():
        cfg["data_csv_path"] = str(Path(_ss_path).resolve())
        print(f"[TFT] Sử dụng CSV từ session_state: {cfg['data_csv_path']}")
    else:
        # Fallback về file mặc định trong thư mục dataset/
        default_csv = repo_root / "dataset" / "2025.csv"
        cfg["data_csv_path"] = str(default_csv.resolve())
        if _ss_path:
            print(f"[TFT] Cảnh báo: session_state['data_path'] = '{_ss_path}' không tồn tại.")
        print(f"[TFT] Sử dụng CSV mặc định (fallback): {cfg['data_csv_path']}")

    run_dir_abs = Path(run_dir).resolve()
    os.makedirs(run_dir_abs, exist_ok=True)
    runtime_conf_path = (run_dir_abs / "air_quality_tft_runtime.yaml").resolve()
    with open(runtime_conf_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    exp_name = f"streamlit_tft_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    selected_locations = normalize_locations(selected_locations)
    if not selected_locations:
        raise ValueError("TFT yêu cầu chọn ít nhất 1 location.")

    cmd = [
        sys.executable,
        "main.py",
        "--exp_name", exp_name,
        "--conf_file_path", str(runtime_conf_path),
        "--seed", str(int(seed)),
        "--locations", ",".join(selected_locations),
    ]

    started = time.time()
    tft_prog = st.progress(0)
    tft_log_box = st.empty()
    tft_log_lines: list[str] = []
    epoch_pat = re.compile(r"Epoch\s+(\d+)\s*/\s*(\d+)", re.IGNORECASE)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    proc = subprocess.Popen(
        cmd,
        cwd=str(tft_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
        env=env,
    )

    stdout_lines: list[str] = []
    if proc.stdout is not None:
        for raw_line in proc.stdout:
            line = raw_line.rstrip("\n")
            stdout_lines.append(line)
            if line.strip():
                tft_log_lines.append(line)
                tft_log_box.code("\n".join(tft_log_lines[-24:]))
            m = epoch_pat.search(line)
            if m:
                ep = int(m.group(1))
                ep_total = int(m.group(2))
                if ep_total > 0:
                    tft_prog.progress(min(ep / ep_total, 1.0))

    proc.wait()
    run_sec = time.time() - started
    proc_stdout = "\n".join(stdout_lines)

    if proc.returncode == 0:
        tft_prog.progress(1.0)

    if proc.returncode != 0:
        raise RuntimeError(
            "TFT chạy lỗi.\n"
            f"Exit code: {proc.returncode}\n"
            f"STDOUT:\n{proc_stdout[-4000:]}"
        )

    # --- Thu thập kết quả ---
    exp_root = tft_root / "log" / cfg["model"] / exp_name
    if not exp_root.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục log TFT: {exp_root}")

    run_dirs = [p for p in exp_root.iterdir() if p.is_dir()]
    if not run_dirs:
        raise FileNotFoundError(f"Không có run folder trong: {exp_root}")
    latest_run_dir = max(run_dirs, key=lambda p: p.stat().st_mtime)

    pred_files = sorted(
        latest_run_dir.glob("transformer_predictions_*.csv"),
        key=lambda p: p.stat().st_mtime,
    )
    if not pred_files:
        raise FileNotFoundError(f"Không tìm thấy file dự đoán TFT trong: {latest_run_dir}")

    pred_parts = []
    for p in pred_files:
        try:
            part = pd.read_csv(p)
            part["_source_file"] = p.name
            pred_parts.append(part)
        except Exception:
            continue
    if not pred_parts:
        raise ValueError("Không đọc được file dự đoán TFT hợp lệ.")
    pred_df = pd.concat(pred_parts, axis=0, ignore_index=True)

    if "location" in pred_df.columns:
        filtered = pred_df[pred_df["location"].astype(str).isin(selected_locations)].copy()
        if not filtered.empty:
            pred_df = filtered

    pred_path = pred_files[-1]
    if not {"actual_aqi", "predicted_aqi"}.issubset(pred_df.columns):
        raise ValueError(f"File dự đoán TFT không đúng format: {pred_path}")

    if "location" not in pred_df.columns:
        pred_df["location"] = "unknown"
    if "time" not in pred_df.columns:
        pred_df["time"] = synthesize_tft_time_from_dataset(repo_root, pred_df["location"])
    else:
        parsed_time = format_time_utc_strings(pred_df["time"])
        if parsed_time.isna().all():
            pred_df["time"] = synthesize_tft_time_from_dataset(repo_root, pred_df["location"])
        else:
            fill_time = synthesize_tft_time_from_dataset(repo_root, pred_df["location"])
            pred_df["time"] = parsed_time.where(parsed_time.notna(), fill_time)

    future_pred_df = pd.DataFrame(
        {
            "time": format_time_utc_strings(pred_df["time"]),
            "location": pred_df["location"].astype(str),
            "predicted": pd.to_numeric(pred_df["predicted_aqi"], errors="coerce"),
        }
    ).dropna(subset=["predicted"])

    y_true = pd.to_numeric(pred_df["actual_aqi"], errors="coerce").dropna().to_numpy(dtype=np.float32)
    y_pred_arr = pd.to_numeric(pred_df["predicted_aqi"], errors="coerce").dropna().to_numpy(dtype=np.float32)
    n = min(len(y_true), len(y_pred_arr))
    if n == 0:
        raise ValueError("Không có dữ liệu dự đoán hợp lệ trong file TFT.")
    y_true, y_pred_arr = y_true[:n], y_pred_arr[:n]

    mse = mean_squared_error(y_true, y_pred_arr)
    last_mae = float(mean_absolute_error(y_true, y_pred_arr))
    last_rmse = float(np.sqrt(mse))
    last_r2 = float(r2_score(y_true, y_pred_arr))

    summary: dict = {
        "model": "tft",
        "device": cfg["device"],
        "seed": int(seed),
        "test_mae": last_mae,
        "test_rmse": last_rmse,
        "test_r2": last_r2,
        "run_sec": float(run_sec),
        "pred_path": str(pred_path),
        "log_dir": str(latest_run_dir),
        "selected_locations": ",".join(selected_locations),
        "stdout_tail": proc_stdout[-2000:],
        "tft_last_test_mae": last_mae,
        "tft_last_test_rmse": last_rmse,
        "tft_last_test_r2": last_r2,
        "train_only_sec": float(run_sec),
    }

    # --- Đọc metrics history ---
    metrics_path = latest_run_dir / "metrics_history.csv"
    if metrics_path.exists():
        history_df = pd.read_csv(metrics_path)
        summary["metrics_path"] = str(metrics_path)
        if "train_sec" in history_df.columns:
            summary["train_only_sec"] = float(
                pd.to_numeric(history_df["train_sec"], errors="coerce").fillna(0).sum()
            )

        required_cols = {"test_loss", "test_mae", "test_rmse", "test_r2"}
        if required_cols.issubset(set(history_df.columns)) and not history_df.empty:
            history_df = history_df.sort_values("epoch").reset_index(drop=True)
            best_idx = pd.to_numeric(history_df["test_loss"], errors="coerce").idxmin()
            best_row = history_df.loc[best_idx]
            last_row = history_df.iloc[-1]

            summary.update(
                {
                    "tft_best_epoch": int(best_row.get("epoch", np.nan)),
                    "tft_best_test_loss": float(best_row.get("test_loss", np.nan)),
                    "tft_best_test_mae": float(best_row.get("test_mae", np.nan)),
                    "tft_best_test_rmse": float(best_row.get("test_rmse", np.nan)),
                    "tft_best_test_r2": float(best_row.get("test_r2", np.nan)),
                    "tft_hist_last_epoch": int(last_row.get("epoch", np.nan)),
                    "tft_hist_last_test_loss": float(last_row.get("test_loss", np.nan)),
                    "tft_hist_last_test_mae": float(last_row.get("test_mae", np.nan)),
                    "tft_hist_last_test_rmse": float(last_row.get("test_rmse", np.nan)),
                    "tft_hist_last_test_r2": float(last_row.get("test_r2", np.nan)),
                    "test_mae": float(best_row.get("test_mae", np.nan)),
                    "test_rmse": float(best_row.get("test_rmse", np.nan)),
                    "test_r2": float(best_row.get("test_r2", np.nan)),
                    "selection": "best_test_loss",
                }
            )
        else:
            summary["selection"] = "last_prediction_file"
    else:
        history_df = pd.DataFrame(
            [
                {
                    "epoch": cfg["num_epochs"],
                    "test_mae": summary["test_mae"],
                    "test_rmse": summary["test_rmse"],
                    "test_r2": summary["test_r2"],
                    "run_sec": summary["run_sec"],
                }
            ]
        )
        summary["metrics_path"] = ""
        summary["selection"] = "last_prediction_file"

    # --- Export artifacts ---
    out_dir = Path(run_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    best_ckpt_candidates = sorted(
        latest_run_dir.glob("*_best.pth"), key=lambda p: p.stat().st_mtime
    )
    if best_ckpt_candidates:
        best_src = best_ckpt_candidates[-1]
        best_dst = out_dir / "best_transformer_tft.pt"
        shutil.copy2(best_src, best_dst)
        summary["model_path"] = str(best_dst)
    else:
        summary["model_path"] = ""

    metrics_dst = out_dir / "metrics_history.csv"
    history_df.to_csv(metrics_dst, index=False)
    summary["metrics_path"] = str(metrics_dst)

    future_path = out_dir / "future_24h_predictions.csv"
    future_pred_df.to_csv(future_path, index=False)
    summary.update(
        {
            "future_pred_path": str(future_path),
            "future_rows": int(len(future_pred_df)),
            "future_locations": int(future_pred_df["location"].nunique()),
            "pred_path": str(future_path),
        }
    )

    return summary, history_df, pred_df
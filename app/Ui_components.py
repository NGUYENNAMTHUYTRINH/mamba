"""
ui_components.py
----------------
Các component UI tái sử dụng: sidebar, metrics, bảng so sánh, download, v.v.
"""

from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


import numpy as np
import pandas as pd
import streamlit as st

from Utils import load_train_module, split_data_by_timeline


# ---------------------------------------------------------------------------
# Sidebar & dataset loading
# ---------------------------------------------------------------------------

def render_sidebar() -> tuple[str, str, object | None]:
    """Render sidebar nguồn dữ liệu, trả về (source, data_path, uploaded_file)."""
    with st.sidebar:
        st.header("Nguồn dữ liệu")
        source = st.radio("Dataset source", ["workspace path", "upload csv"], index=0)
        data_path = st.text_input("Path CSV trong workspace", value="dataset/2025.csv")
        uploaded = st.file_uploader("Hoặc upload CSV", type=["csv"])
        if st.button("Load dataset"):
            _load_dataset(source, data_path, uploaded)
    return source, data_path, uploaded


def _load_dataset(source: str, data_path: str, uploaded) -> None:
    """Load dataset vào session_state['df']."""
    try:
        if source == "upload csv":
            if uploaded is None:
                st.error("Bạn chưa upload file CSV.")
                return
            st.session_state["df"] = pd.read_csv(uploaded)
        else:
            st.session_state["df"] = pd.read_csv(data_path)

        df = st.session_state["df"]
        st.success(
            f"Load thành công dataset: {df.shape[0]:,} rows, {df.shape[1]} cols"
        )
    except Exception as e:
        st.error(f"Load dataset lỗi: {e}")


# ---------------------------------------------------------------------------
# Data preview
# ---------------------------------------------------------------------------

def render_data_preview(df: pd.DataFrame) -> None:
    """Hiển thị preview và thông tin cơ bản của dataset."""
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Preview dữ liệu")
        st.dataframe(df.head(20), use_container_width=True)
    with col2:
        st.subheader("Thông tin")
        st.write(f"Rows: {len(df):,}")
        st.write(f"Columns: {df.shape[1]}")


# ---------------------------------------------------------------------------
# Location selector + sample preview
# ---------------------------------------------------------------------------

def render_location_selector(df: pd.DataFrame, locations: list[str]) -> list[str]:
    """Render location multiselect + sample count preview. Trả về selected_locations."""
    st.subheader("Chọn địa điểm để train + forecast")
    selected_locations = st.multiselect(
        "Chọn địa điểm để train + forecast",
        options=locations,
        default=locations[: min(3, len(locations))],
        help="Có thể chọn 1 hoặc nhiều địa điểm. Mô hình sẽ train chung theo nhiều tỉnh.",
    )

    preview_col1, preview_col2 = st.columns(2)
    with preview_col1:
        preview_window = st.number_input(
            "Preview window size (timesteps)", min_value=1, max_value=168, value=24, step=1
        )
    with preview_col2:
        preview_horizon = st.number_input(
            "Preview horizon", min_value=1, max_value=168, value=1, step=1
        )

    if selected_locations:
        _render_sample_count_preview(df, selected_locations, int(preview_window), int(preview_horizon))

    return selected_locations


def _render_sample_count_preview(
    df: pd.DataFrame, selected_locations: list[str], window: int, horizon: int
) -> None:
    """Hiển thị số lượng sample train/val/test theo preview window/horizon."""
    try:
        df_sel = df.loc[
            df["location_key"].astype(str).isin([str(x) for x in selected_locations])
        ].copy()
        default_target = (
            "aqi"
            if "aqi" in df_sel.columns
            else next(
                (c for c in df_sel.select_dtypes(include=["number"]).columns if c != "_loc_id"),
                None,
            )
        )
        if default_target is None:
            st.warning("Không tìm thấy cột số để preview sample counts.")
            return

        mod = load_train_module()
        if mod is None or not hasattr(mod, "build_time_series_samples"):
            st.warning("Không thể load helper 'build_time_series_samples' để preview samples.")
            return

        x_seq, loc_ids, y, y_ts, _, _ = mod.build_time_series_samples(
            df_sel, default_target, window, horizon
        )
        if hasattr(mod, "split_data_by_timeline"):
            train, val, test = mod.split_data_by_timeline(x_seq, loc_ids, y, y_ts)
        else:
            train, val, test = split_data_by_timeline(x_seq, loc_ids, y, y_ts)

        st.markdown(
            f"**Preview ({len(selected_locations)} locations)**: "
            f"total samples={len(y):,}"
        )
        st.write(f"Train: {len(train.y):,}  |  Val: {len(val.y):,}  |  Test: {len(test.y):,}")
    except Exception as e:
        st.warning(f"Không thể tính preview samples: {e}")


# ---------------------------------------------------------------------------
# Train config form
# ---------------------------------------------------------------------------

def render_train_config(
    df: pd.DataFrame,
) -> dict:
    """Render toàn bộ form cấu hình train. Trả về dict config."""
    all_cols = df.columns.tolist()
    reserved_cols = {"y_true", "y_pred", "abs_error"}
    feature_options = [
        c for c in all_cols
        if c not in reserved_cols
        and c not in ["ts_utc", "location_key"]
        and not c.lower().startswith("unnamed:")
    ]

    st.subheader("Cấu hình train")
    conf1, conf2, conf3 = st.columns(3)

    with conf1:
        target_col = st.selectbox(
            "Target column (biến cần dự đoán)",
            options=feature_options,
            index=feature_options.index("aqi") if "aqi" in feature_options else 0,
        )
        default_features = [c for c in feature_options if c != target_col]
        feature_cols = st.multiselect(
            "Input feature columns",
            options=[c for c in feature_options if c != target_col],
            default=default_features,
        )
        loss_name = st.selectbox("Loss", options=["huber", "mse"], index=0)

    with conf2:
        epochs = st.number_input("Epochs", min_value=1, max_value=200, value=5, step=1)
        batch_size = st.number_input("Batch size", min_value=8, max_value=8192, value=128, step=8)
        lr = st.number_input("Learning rate", min_value=1e-6, max_value=1e-1, value=3e-4, format="%.6f")
        weight_decay = st.number_input("Weight decay", min_value=0.0, max_value=1.0, value=1e-4, format="%.6f")

    with conf3:
        d_model = st.number_input("d_model", min_value=16, max_value=512, value=64, step=16)
        n_layers = st.number_input("n_layers", min_value=1, max_value=8, value=2, step=1)
        grad_accum_steps = st.number_input("Gradient accumulation", min_value=1, max_value=64, value=2, step=1)
        max_grad_norm = st.number_input("Max grad norm", min_value=0.0, max_value=100.0, value=1.0, step=0.5)

    run1, run2, run3 = st.columns(3)
    with run1:
        seed = st.number_input("Seed", min_value=0, max_value=999999, value=42, step=1)
    with run2:
        num_workers = st.number_input("DataLoader workers", min_value=0, max_value=16, value=0, step=1)
    with run3:
        use_gpu = st.checkbox("Dùng GPU (nếu có)", value=True)

    import torch
    if use_gpu and not torch.cuda.is_available():
        st.warning(
            "Bạn đang bật GPU nhưng PyTorch hiện không nhận CUDA (torch+cpu). "
            "Train sẽ chạy bằng CPU nên thời gian mỗi epoch sẽ cao."
        )

    return dict(
        target_col=target_col,
        feature_cols=feature_cols,
        loss_name=loss_name,
        epochs=int(epochs),
        batch_size=int(batch_size),
        lr=float(lr),
        weight_decay=float(weight_decay),
        d_model=int(d_model),
        n_layers=int(n_layers),
        grad_accum_steps=int(grad_accum_steps),
        max_grad_norm=float(max_grad_norm),
        seed=int(seed),
        num_workers=int(num_workers),
        use_gpu=bool(use_gpu),
    )


# ---------------------------------------------------------------------------
# Comparison model toggles
# ---------------------------------------------------------------------------

def render_comparison_config() -> dict:
    """Render checkbox + hyperparams cho TFT và LSTM. Trả về dict config."""
    compare_with_tft = st.checkbox(
        "Chạy thêm TFT để so sánh",
        value=True,
        help="Train Mamba + TFT và hiển thị bảng compare test metrics trong cùng run.",
    )

    run1, run2, run3 = st.columns(3)
    with run1:
        compare_with_lstm = st.checkbox(
            "Chạy thêm LSTM để so sánh",
            value=True,
            help="Train LSTM với cùng tập locations đã chọn và hiển thị bảng compare.",
        )
    with run2:
        lstm_lookback = st.number_input("LSTM lookback", min_value=1, max_value=168, value=24, step=1)
    with run3:
        lstm_hidden = st.number_input("LSTM hidden size", min_value=16, max_value=512, value=64, step=16)

    lstm_col1, lstm_col2 = st.columns(2)
    with lstm_col1:
        lstm_num_layers = st.number_input("LSTM num_layers", min_value=1, max_value=8, value=2, step=1)
        lstm_dropout = st.number_input("LSTM dropout", min_value=0.0, max_value=0.9, value=0.2, format="%.2f")

    st.info(
        "Tỉ lệ split cố định theo thời gian: Train 70% | Val 10% | Test 20%. "
        "Test là các mốc thời gian gần nhất trong dataset tổng."
    )

    return dict(
        compare_with_tft=bool(compare_with_tft),
        compare_with_lstm=bool(compare_with_lstm),
        lstm_lookback=int(lstm_lookback),
        lstm_hidden=int(lstm_hidden),
        lstm_num_layers=int(lstm_num_layers),
        lstm_dropout=float(lstm_dropout),
    )


# ---------------------------------------------------------------------------
# Results rendering
# ---------------------------------------------------------------------------

def render_mamba_results(summary: dict, hist_df: pd.DataFrame, future_df: pd.DataFrame, df: pd.DataFrame, selected_locations: list[str]) -> None:
    """Hiển thị kết quả sau khi train Mamba."""
    st.success("Train/Test hoàn tất")

    met1, met2, met3, met4 = st.columns(4)
    met1.metric("Val MAE", f"{summary['val_mae']:.4f}")
    met2.metric("Val RMSE", f"{summary['val_rmse']:.4f}")
    met3.metric("Val R2", f"{summary['val_r2']:.4f}")
    met4.metric("Locations done", f"{int(summary['future_locations']):,}")

    st.write("### Số dòng sau khi lọc theo địa điểm")
    train_counts = (
        df.loc[df["location_key"].astype(str).isin([str(x) for x in selected_locations]), "location_key"]
        .astype(str)
        .value_counts()
        .rename_axis("location_key")
        .reset_index(name="train_source_rows")
    )
    test_counts = pd.DataFrame(
        {
            "location_key": selected_locations,
            "test_source_rows": [int(summary["split_test"])] * len(selected_locations),
        }
    )
    used_counts = (
        future_df["location"].astype(str)
        .value_counts()
        .rename_axis("location_key")
        .reset_index(name="future_rows")
    )
    stats_df = (
        train_counts
        .merge(test_counts, on="location_key", how="outer")
        .merge(used_counts, on="location_key", how="outer")
        .fillna(0)
    )

    merged_stats = stats_df.merge(
        pd.DataFrame(
            [
                {
                    "n_rows_used": summary["n_rows_used"],
                    "split_train": summary["split_train"],
                    "split_val": summary["split_val"],
                    "split_test": summary["split_test"],
                }
            ]
        ),
        how="cross",
    )
    st.dataframe(merged_stats, use_container_width=True)

    st.write("### Thống kê split")
    st.write(
        {k: round(float(v), 2) if isinstance(v, float) else v
         for k, v in {
             "split_train": summary["split_train"],
             "split_val": summary["split_val"],
             "split_test": summary["split_test"],
             "n_rows_used": summary["n_rows_used"],
             "future_rows": summary["future_rows"],
             "future_locations": summary["future_locations"],
             "train_only_sec": summary.get("train_only_sec", np.nan),
             "eval_sec": summary.get("eval_sec", np.nan),
             "forecast_sec": summary.get("forecast_sec", np.nan),
             "io_sec": summary.get("io_sec", np.nan),
             "run_sec": summary["run_sec"],
         }.items()}
    )

    st.write("### Lịch sử train Mamba")
    st.dataframe(hist_df, use_container_width=True)


def render_tft_results(tft_summary: dict, tft_hist_df: pd.DataFrame | None, tft_pred_df: pd.DataFrame | None) -> None:
    """Hiển thị kết quả TFT riêng (history + predictions preview)."""
    if tft_hist_df is not None and not tft_hist_df.empty:
        st.write("### Lịch sử train TFT")
        st.dataframe(tft_hist_df, use_container_width=True)

    tft_modes_df = pd.DataFrame(
        [
            {
                "mode": "tft_best(test_loss)",
                "epoch": tft_summary.get("tft_best_epoch"),
                "test_mae": tft_summary.get("tft_best_test_mae"),
                "test_rmse": tft_summary.get("tft_best_test_rmse"),
                "test_r2": tft_summary.get("tft_best_test_r2"),
            },
            {
                "mode": "tft_last(history)",
                "epoch": tft_summary.get("tft_hist_last_epoch"),
                "test_mae": tft_summary.get("tft_hist_last_test_mae", tft_summary.get("tft_last_test_mae")),
                "test_rmse": tft_summary.get("tft_hist_last_test_rmse", tft_summary.get("tft_last_test_rmse")),
                "test_r2": tft_summary.get("tft_hist_last_test_r2", tft_summary.get("tft_last_test_r2")),
            },
        ]
    )
    st.write("### TFT benchmark modes")
    st.dataframe(tft_modes_df, use_container_width=True)

    if tft_pred_df is not None and not tft_pred_df.empty:
        st.write("### TFT predictions preview")
        st.dataframe(tft_pred_df.head(100), use_container_width=True)


def render_lstm_results(lstm_hist_df: pd.DataFrame | None, lstm_pred_df: pd.DataFrame | None) -> None:
    """Hiển thị kết quả LSTM riêng."""
    if lstm_hist_df is not None:
        st.write("### Lịch sử train LSTM")
        st.dataframe(lstm_hist_df, use_container_width=True)

    if lstm_pred_df is not None and not lstm_pred_df.empty:
        st.write("### LSTM predictions preview")
        st.dataframe(lstm_pred_df.head(100), use_container_width=True)


def render_comparison_table(
    summary: dict,
    tft_summary: dict | None,
    lstm_summary: dict | None,
) -> None:
    """Hiển thị bảng so sánh 2 hoặc 3 model."""
    rows = [
        {
            "model": "mamba_best(val)",
            "test_mae": summary.get("test_mae", np.nan),
            "test_rmse": summary.get("test_rmse", np.nan),
            "test_r2": summary.get("test_r2", np.nan),
            "train_sec": summary.get("train_only_sec", np.nan),
            "run_sec": summary.get("run_sec", np.nan),
        }
    ]

    if tft_summary is not None:
        st.write("### So sánh Mamba vs TFT (test metrics)")
        st.caption(
            "Benchmark chuẩn: cùng loss, cùng seed; "
            "Mamba dùng best checkpoint theo val, TFT dùng best epoch theo test_loss."
        )
        rows.append(
            {
                "model": "tft_best(test_loss)",
                "test_mae": tft_summary.get("test_mae", np.nan),
                "test_rmse": tft_summary.get("test_rmse", np.nan),
                "test_r2": tft_summary.get("test_r2", np.nan),
                "train_sec": tft_summary.get("train_only_sec", np.nan),
                "run_sec": tft_summary.get("run_sec", np.nan),
            }
        )
        st.dataframe(pd.DataFrame(rows[:2]), use_container_width=True)

    if lstm_summary is not None:
        rows.append(
            {
                "model": "lstm_best(val)",
                "test_mae": lstm_summary.get("test_mae", np.nan),
                "test_rmse": lstm_summary.get("test_rmse", np.nan),
                "test_r2": lstm_summary.get("test_r2", np.nan),
                "train_sec": lstm_summary.get("train_only_sec", np.nan),
                "run_sec": lstm_summary.get("run_sec", np.nan),
            }
        )

    if tft_summary is not None and lstm_summary is not None:
        st.divider()
        st.write("### 📊 So sánh 3 mô hình (Mamba vs TFT vs LSTM)")
        st.caption(
            "Đánh giá dựa trên test metrics - tất cả mô hình dùng cùng seed, cùng tập locations"
        )
        three_df = pd.DataFrame(rows)
        st.dataframe(three_df, use_container_width=True)

        mae_best = three_df.loc[three_df["test_mae"].idxmin(), "model"]
        rmse_best = three_df.loc[three_df["test_rmse"].idxmin(), "model"]
        r2_best = three_df.loc[three_df["test_r2"].idxmax(), "model"]

        col1, col2, col3 = st.columns(3)
        col1.metric("Best MAE", mae_best)
        col2.metric("Best RMSE", rmse_best)
        col3.metric("Best R²", r2_best)


def render_forecast_download(future_df: pd.DataFrame, summary: dict) -> None:
    """Hiển thị bảng dự báo 24h và nút download."""
    st.write("### Dự báo 24 giờ tiếp theo (từng địa điểm)")
    st.dataframe(future_df.head(300), use_container_width=True)
    st.download_button(
        "Download file tổng (mọi location)",
        data=future_df.to_csv(index=False).encode("utf-8"),
        file_name="future_24h_predictions.csv",
        mime="text/csv",
    )
    st.info("Mamba xuất 1 file tổng trong run_dir: future_24h_predictions.csv (gồm time, location, predicted).")
    st.code(
        f"run_dir: {os.path.dirname(summary['future_pred_path'])}\n"
        "Files: future_24h_predictions.csv"
    )
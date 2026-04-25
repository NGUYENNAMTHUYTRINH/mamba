"""
main.py
-------
Entry point của Streamlit app.
Chạy bằng:  streamlit run app/main.py
"""

from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime

import pandas as pd
import streamlit as st

from Pipeline_lstm import run_lstm_pipeline
from Pipeline_mamba import train_pipeline
from Pipeline_tft import run_tft_pipeline
from Ui_components import (
    render_comparison_config,
    render_comparison_table,
    render_data_preview,
    render_forecast_download,
    render_location_selector,
    render_lstm_results,
    render_mamba_results,
    render_sidebar,
    render_tft_results,
    render_train_config,
)


def main() -> None:
    st.set_page_config(page_title="Mamba Trainer", layout="wide")
    st.title("Mamba Train/Val/Test Dashboard")
    st.caption(
        "Chọn dataset, chọn target + input features, train/val/test 70/10/20, "
        "và xem log + dự đoán test."
    )

    # --- Sidebar / load dataset ---
    render_sidebar()

    if "df" not in st.session_state:
        st.session_state["df"] = None

    df: pd.DataFrame | None = st.session_state["df"]
    if df is None:
        st.info("Hãy bấm 'Load dataset' để bắt đầu.")
        return

    # --- Preview ---
    render_data_preview(df)

    # --- Validate ---
    if "location_key" not in df.columns:
        st.error("Dataset cần có cột location_key để tách train/dự báo theo từng địa điểm.")
        return

    locations = sorted(df["location_key"].dropna().astype(str).unique().tolist())
    if not locations:
        st.error("Không tìm thấy location_key hợp lệ trong dataset.")
        return

    # --- Location selector ---
    selected_locations = render_location_selector(df, locations)

    # --- Train config ---
    train_cfg = render_train_config(df)

    # --- Comparison config ---
    cmp_cfg = render_comparison_config()

    # --- Train button ---
    if st.button("Train & Test", type="primary"):
        _run_training(df, selected_locations, train_cfg, cmp_cfg)


def _run_training(
    df: pd.DataFrame,
    selected_locations: list[str],
    train_cfg: dict,
    cmp_cfg: dict,
) -> None:
    """Validate inputs và chạy toàn bộ pipeline train."""
    if not train_cfg["feature_cols"]:
        st.error("Bạn cần chọn ít nhất 1 cột input.")
        return
    if not selected_locations:
        st.error("Bạn cần chọn ít nhất 1 location.")
        return

    with st.spinner("Đang train và evaluate..."):
        try:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            mamba_run_dir = os.path.join("outputs", "mamba_runs", run_id)
            tft_run_dir = os.path.join("outputs", "transformers_runs", run_id)
            lstm_run_dir = os.path.join("outputs", "lstm_runs", run_id)
            for d in [mamba_run_dir, tft_run_dir, lstm_run_dir]:
                os.makedirs(d, exist_ok=True)

            # ── Mamba ──────────────────────────────────────────────────────
            summary, hist_df, future_df = train_pipeline(
                df=df,
                forecast_base_df=None,
                selected_locations=selected_locations,
                target_col=train_cfg["target_col"],
                feature_cols=train_cfg["feature_cols"],
                epochs=train_cfg["epochs"],
                batch_size=train_cfg["batch_size"],
                lr=train_cfg["lr"],
                weight_decay=train_cfg["weight_decay"],
                d_model=train_cfg["d_model"],
                n_layers=train_cfg["n_layers"],
                loss_name=train_cfg["loss_name"],
                seed=train_cfg["seed"],
                num_workers=train_cfg["num_workers"],
                use_gpu=train_cfg["use_gpu"],
                log_interval=50,
                grad_accum_steps=train_cfg["grad_accum_steps"],
                max_grad_norm=train_cfg["max_grad_norm"],
                run_dir=mamba_run_dir,
            )

            # ── TFT (optional) ─────────────────────────────────────────────
            tft_summary, tft_hist_df, tft_pred_df = None, None, None
            if cmp_cfg["compare_with_tft"]:
                with st.spinner("Đang chạy TFT để so sánh..."):
                    tft_summary, tft_hist_df, tft_pred_df = run_tft_pipeline(
                        selected_locations=selected_locations,
                        epochs=train_cfg["epochs"],
                        batch_size=train_cfg["batch_size"],
                        lr=train_cfg["lr"],
                        weight_decay=train_cfg["weight_decay"],
                        loss_name=train_cfg["loss_name"],
                        seed=train_cfg["seed"],
                        use_gpu=train_cfg["use_gpu"],
                        run_dir=tft_run_dir,
                    )

            # ── LSTM (optional) ────────────────────────────────────────────
            lstm_summary, lstm_hist_df, lstm_pred_df = None, None, None
            if cmp_cfg["compare_with_lstm"]:
                with st.spinner("Đang chạy LSTM để so sánh..."):
                    lstm_summary, lstm_hist_df, lstm_pred_df = run_lstm_pipeline(
                        df=df,
                        selected_locations=selected_locations,
                        target_col=train_cfg["target_col"],
                        feature_cols=train_cfg["feature_cols"],
                        lookback=cmp_cfg["lstm_lookback"],
                        horizon=1,
                        epochs=train_cfg["epochs"],
                        batch_size=train_cfg["batch_size"],
                        lr=train_cfg["lr"],
                        hidden_size=cmp_cfg["lstm_hidden"],
                        num_layers=cmp_cfg["lstm_num_layers"],
                        dropout=cmp_cfg["lstm_dropout"],
                        seed=train_cfg["seed"],
                        use_gpu=train_cfg["use_gpu"],
                        run_dir=lstm_run_dir,
                    )

        except Exception as e:
            st.error(f"Train/Test lỗi: {e}")
            return

    # ── Render results ─────────────────────────────────────────────────────
    render_mamba_results(summary, hist_df, future_df, df, selected_locations)

    if tft_summary is not None:
        render_tft_results(tft_summary, tft_hist_df, tft_pred_df)

    if lstm_summary is not None:
        render_lstm_results(lstm_hist_df, lstm_pred_df)

    render_comparison_table(summary, tft_summary, lstm_summary)
    render_forecast_download(future_df, summary)


if __name__ == "__main__":
    main()
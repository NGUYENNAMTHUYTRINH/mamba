"""
main.py
-------
Entry point của Streamlit app.
Chạy bằng:  streamlit run app/Main.py
"""

from __future__ import annotations

import os
import sys

# Đảm bảo nhận diện đúng các module trong thư mục app
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime
import pandas as pd
import streamlit as st

# Import các pipeline huấn luyện
from Pipeline_lstm import run_lstm_pipeline
from Pipeline_mamba import train_pipeline
from Pipeline_tft import run_tft_pipeline

# Import các thành phần giao diện
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
    st.set_page_config(page_title="AQI Model Comparison", layout="wide")
    st.title("🚀 AQI Forecasting: Mamba vs TFT vs LSTM")
    st.caption(
        "Hệ thống so sánh hiệu năng các kiến trúc Deep Learning trong dự báo chất lượng không khí."
    )

    # --- 1. Sidebar & Load Dataset ---
    render_sidebar()

    if "df" not in st.session_state:
        st.info("👈 Vui lòng load dataset từ sidebar để bắt đầu.")
        return

    df = st.session_state["df"]

    # --- 2. Cấu hình chung và chọn địa điểm ---
    selected_locations = render_location_selector(df)
    
    col_cfg1, col_cfg2 = st.columns(2)
    with col_cfg1:
        train_cfg = render_train_config()
    with col_cfg2:
        # Nhận cấu hình so sánh (bao gồm các checkbox run_mamba, run_tft, run_lstm)
        cmp_cfg = render_comparison_config()

    # Preview dữ liệu (đã bọc try-except bên trong component)
    render_data_preview(df)

    # --- 3. Khởi tạo các biến chứa kết quả (Tránh lỗi NameError) ---
    # Mamba
    summary, hist_df, future_df = None, None, None
    # TFT
    tft_summary, tft_hist_df, tft_pred_df = None, None, None
    # LSTM
    lstm_summary, lstm_hist_df, lstm_pred_df = None, None, None

    # Tạo đường dẫn lưu kết quả dựa trên thời gian
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_run_dir = os.path.join("runs", timestamp)

    # --- 4. Vòng lặp huấn luyện khi nhấn nút ---
    if st.button("🔥 Run Training", use_container_width=True):
        if not (cmp_cfg["run_mamba"] or cmp_cfg["run_tft"] or cmp_cfg["run_lstm"]):
            st.warning("⚠️ Vui lòng tích chọn ít nhất một mô hình ở trên trước khi chạy!")
            st.stop()

        if not selected_locations:
            st.warning("⚠️ Vui lòng chọn ít nhất một địa điểm.")
            st.stop()

        if not train_cfg["feature_cols"]:
            st.warning("⚠️ Vui lòng chọn ít nhất một cột feature.")
            st.stop()

        status_placeholder = st.empty()

        try:
            # A. MAMBA — chỉ chạy khi được tích chọn
            if cmp_cfg["run_mamba"]:
                status_placeholder.info("⏳ [1/3] Đang huấn luyện Mamba...")
                mamba_run_dir = os.path.join(base_run_dir, "mamba")
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
                    d_model=cmp_cfg["mamba_d_model"],
                    n_layers=cmp_cfg["mamba_n_layers"],
                    loss_name=train_cfg["loss_name"],
                    seed=train_cfg["seed"],
                    num_workers=train_cfg["num_workers"],
                    use_gpu=train_cfg["use_gpu"],
                    log_interval=50,
                    grad_accum_steps=train_cfg["grad_accum_steps"],
                    max_grad_norm=train_cfg["max_grad_norm"],
                    run_dir=mamba_run_dir,
                )

            # B. TFT — chỉ chạy khi được tích chọn
            if cmp_cfg["run_tft"]:
                status_placeholder.info("⏳ [2/3] Đang huấn luyện TFT...")
                tft_run_dir = os.path.join(base_run_dir, "tft")
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

            # C. LSTM — chỉ chạy khi được tích chọn
            if cmp_cfg["run_lstm"]:
                status_placeholder.info("⏳ [3/3] Đang huấn luyện LSTM...")
                lstm_run_dir = os.path.join(base_run_dir, "lstm")
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

            models_done = sum([
                summary is not None,
                tft_summary is not None,
                lstm_summary is not None,
            ])
            status_placeholder.success(f"✅ Hoàn thành huấn luyện {models_done} mô hình!")

        except Exception as e:
            st.error(f"❌ Quá trình huấn luyện gặp lỗi: {e}")
            import traceback
            traceback.print_exc()
            return

    # --- 5. Hiển thị kết quả (Render Results) ---
    # Chỉ hiển thị nếu biến kết quả không phải None (nghĩa là đã được chạy thành công)
    
    # Kết quả Mamba
    if summary is not None:
        render_mamba_results(summary, hist_df, future_df, df, selected_locations)
        # Nút tải dự báo cho Mamba (nếu có)
        if future_df is not None:
            render_forecast_download(future_df, summary)

    # Kết quả TFT
    if tft_summary is not None:
        render_tft_results(tft_summary, tft_hist_df, tft_pred_df)

    # Kết quả LSTM
    if lstm_summary is not None:
        render_lstm_results(lstm_hist_df, lstm_pred_df)

    # --- 6. Bảng so sánh tổng hợp ---
    # Hàm này nên được thiết kế để nhận diện cái nào None thì bỏ qua trong bảng
    render_comparison_table(summary, tft_summary, lstm_summary)


if __name__ == "__main__":
    main()
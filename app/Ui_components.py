"""
ui_components.py
----------------
Các component UI tái sử dụng: sidebar, metrics, bảng so sánh, download, v.v.
"""

from __future__ import annotations

import os
import sys
import shutil
import tempfile
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


import numpy as np
import pandas as pd
import streamlit as st

from Utils import load_train_module, split_data_by_timeline

# Thư mục lưu tạm file upload (nằm cạnh app/)
_UPLOAD_TEMP_DIR = Path(__file__).parent.parent / "runs" / "uploaded"


def _fmt_metric(value, fallback=np.nan) -> str:
    try:
        if value is None:
            value = fallback
        return f"{float(value):.4f}"
    except Exception:
        return f"{float(fallback):.4f}"


# ---------------------------------------------------------------------------
# Sidebar & dataset loading
# ---------------------------------------------------------------------------

def render_sidebar() -> tuple[str, str, object | None]:
    """Render sidebar nguồn dữ liệu, trả về (source, data_path, uploaded_file)."""
    with st.sidebar:
        st.header("📂 Nguồn dữ liệu")

        # ── Trạng thái dataset hiện tại ─────────────────────────────────────
        _active_path = st.session_state.get("data_path", None)
        _active_df   = st.session_state.get("df", None)
        if _active_df is not None and _active_path:
            st.success(
                f"✅ Dataset đang dùng:\n"
                f"`{Path(_active_path).name}`\n"
                f"{_active_df.shape[0]:,} rows · {_active_df.shape[1]} cols"
            )
        elif _active_df is not None:
            st.info(
                f"📄 Dataset đang dùng: *(file upload tạm)*\n"
                f"{_active_df.shape[0]:,} rows · {_active_df.shape[1]} cols"
            )
        else:
            st.warning("⚠️ Chưa load dataset. Hãy chọn nguồn bên dưới và nhấn **Load**.")

        st.divider()

        # ── Chọn nguồn ──────────────────────────────────────────────────────
        source = st.radio(
            "Nguồn dataset",
            ["📁 Đường dẫn trong workspace", "⬆️ Upload file CSV"],
            index=0,
            help="Chọn cách cung cấp file CSV đầu vào cho tất cả các model (Mamba, TFT, LSTM).",
        )
        use_upload = source.startswith("⬆️")

        data_path = ""
        uploaded  = None

        if not use_upload:
            data_path = st.text_input(
                "Path CSV (tương đối hoặc tuyệt đối)",
                value=st.session_state.get("_sidebar_path_input", "dataset/2025.csv"),
                help="Ví dụ: dataset/2025.csv  hoặc  D:/data/myfile.csv",
                key="_sidebar_path_input",
            )
            st.caption(f"📌 Thư mục gốc: `{Path.cwd()}`")
        else:
            uploaded = st.file_uploader(
                "Chọn file CSV để upload",
                type=["csv"],
                help="File sẽ được lưu tạm vào thư mục `runs/uploaded/` để TFT có thể đọc.",
            )
            if uploaded is not None:
                st.caption(f"📎 File đã chọn: **{uploaded.name}** ({uploaded.size / 1024:.1f} KB)")

        # ── Nút Load / Reset ─────────────────────────────────────────────────
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            load_clicked = st.button("⬇️ Load", use_container_width=True, type="primary")
        with btn_col2:
            reset_clicked = st.button("🗑️ Reset", use_container_width=True)

        if reset_clicked:
            for key in ["df", "data_path", "_upload_saved_path"]:
                st.session_state.pop(key, None)
            st.rerun()

        if load_clicked:
            _load_dataset(use_upload, data_path, uploaded)

    return source, data_path, uploaded


def _load_dataset(use_upload: bool, data_path: str, uploaded) -> None:
    """Load dataset vào session_state['df'] và lưu path tuyệt đối vào session_state['data_path']."""
    try:
        if use_upload:
            # ── Trường hợp upload file ──────────────────────────────────────
            if uploaded is None:
                st.sidebar.error("❌ Bạn chưa chọn file CSV để upload.")
                return

            # Lưu file tạm ra disk để TFT subprocess đọc được đường dẫn thực
            _UPLOAD_TEMP_DIR.mkdir(parents=True, exist_ok=True)
            saved_path = _UPLOAD_TEMP_DIR / uploaded.name
            with open(saved_path, "wb") as f:
                f.write(uploaded.getbuffer())

            df = pd.read_csv(saved_path)
            st.session_state["df"]              = df
            st.session_state["data_path"]       = str(saved_path.resolve())
            st.session_state["_upload_saved_path"] = str(saved_path.resolve())

            st.sidebar.success(
                f"✅ Upload thành công: **{uploaded.name}**\n"
                f"{df.shape[0]:,} rows · {df.shape[1]} cols"
            )

        else:
            # ── Trường hợp nhập đường dẫn ───────────────────────────────────
            if not data_path or not data_path.strip():
                st.sidebar.error("❌ Đường dẫn CSV không được để trống.")
                return

            abs_path = str(Path(data_path.strip()).resolve())
            if not Path(abs_path).exists():
                st.sidebar.error(
                    f"❌ Không tìm thấy file:\n`{abs_path}`\n\n"
                    "Hãy kiểm tra lại đường dẫn hoặc dùng đường dẫn tuyệt đối."
                )
                return

            df = pd.read_csv(abs_path)
            st.session_state["df"]        = df
            st.session_state["data_path"] = abs_path

            st.sidebar.success(
                f"✅ Load thành công: **{Path(abs_path).name}**\n"
                f"{df.shape[0]:,} rows · {df.shape[1]} cols"
            )

    except Exception as e:
        st.sidebar.error(f"❌ Load dataset lỗi: {e}")


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

def render_location_selector(df: pd.DataFrame) -> list[str]:
    """Render location multiselect + sample count preview. Trả về selected_locations."""
    locations = sorted(df["location_key"].dropna().astype(str).unique().tolist()) if "location_key" in df.columns else []
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

def render_train_config() -> dict:
    """Render toàn bộ form cấu hình train. Trả về dict config."""
    df = st.session_state.get("df", pd.DataFrame())
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
        batch_size = st.number_input("Batch size", min_value=8, max_value=8192, value=512, step=8)
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
    """Render checkbox chọn model + hyperparams riêng cho từng model."""
    st.subheader("Chọn mô hình để huấn luyện")
    st.info(
        "Tích chọn một hoặc nhiều mô hình. "
        "Tỉ lệ split cố định: Train 70% | Val 10% | Test 20%."
    )

    # ── Checkbox 3 model ────────────────────────────────────────────────────
    chk1, chk2, chk3 = st.columns(3)
    with chk1:
        run_mamba = st.checkbox("🟣 Mamba", value=True,
                                help="Huấn luyện Mamba SSM.")
    with chk2:
        run_tft = st.checkbox("🔵 TFT (Transformer)", value=False,
                              help="Huấn luyện Temporal Fusion Transformer.")
    with chk3:
        run_lstm = st.checkbox("🟢 LSTM", value=False,
                               help="Huấn luyện LSTM baseline.")

    # ── Hyperparams Mamba (chỉ hiện khi được chọn) ─────────────────────────
    mamba_lookback, mamba_d_model, mamba_n_layers = 24, 64, 2
    if run_mamba:
        with st.expander("⚙️ Cấu hình Mamba", expanded=False):
            mc1, mc2, mc3 = st.columns(3)
            with mc1:
                mamba_lookback = st.number_input("Lookback (timesteps)", min_value=1, max_value=168, value=24, step=1, key="mamba_lookback")
            with mc2:
                mamba_d_model = st.number_input("d_model", min_value=16, max_value=512, value=64, step=16, key="mamba_d_model")
            with mc3:
                mamba_n_layers = st.number_input("n_layers", min_value=1, max_value=8, value=2, step=1, key="mamba_n_layers")

    # ── Hyperparams TFT (chỉ hiện khi được chọn) ───────────────────────────
    tft_lookback, tft_hidden = 24, 64
    if run_tft:
        with st.expander("⚙️ Cấu hình TFT", expanded=False):
            tc1, tc2 = st.columns(2)
            with tc1:
                tft_lookback = st.number_input("Lookback (timesteps)", min_value=1, max_value=168, value=24, step=1, key="tft_lookback")
            with tc2:
                tft_hidden = st.number_input("Hidden size", min_value=16, max_value=512, value=64, step=16, key="tft_hidden")

    # ── Hyperparams LSTM (chỉ hiện khi được chọn) ──────────────────────────
    lstm_lookback, lstm_hidden, lstm_num_layers, lstm_dropout = 24, 256, 4, 0.3
    if run_lstm:
        with st.expander("⚙️ Cấu hình LSTM", expanded=False):
            lc1, lc2, lc3, lc4 = st.columns(4)
            with lc1:
                lstm_lookback = st.number_input("Lookback", min_value=1, max_value=168, value=24, step=1, key="lstm_lookback")
            with lc2:
                lstm_hidden = st.number_input("Hidden size", min_value=16, max_value=512, value=64, step=16, key="lstm_hidden")
            with lc3:
                lstm_num_layers = st.number_input("Num layers", min_value=1, max_value=8, value=2, step=1, key="lstm_num_layers")
            with lc4:
                lstm_dropout = st.number_input("Dropout", min_value=0.0, max_value=0.9, value=0.2, format="%.2f", key="lstm_dropout")

    return dict(
        run_mamba=bool(run_mamba),
        run_tft=bool(run_tft),
        run_lstm=bool(run_lstm),
        mamba_lookback=int(mamba_lookback),
        mamba_d_model=int(mamba_d_model),
        mamba_n_layers=int(mamba_n_layers),
        tft_lookback=int(tft_lookback),
        tft_hidden=int(tft_hidden),
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
    met1.metric("Val MAE (norm)", _fmt_metric(summary.get("val_mae_norm")))
    met2.metric("Val RMSE (norm)", _fmt_metric(summary.get("val_rmse_norm")))
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
    met1, met2, met3 = st.columns(3)
    met1.metric("Test MAE (norm)", _fmt_metric(tft_summary.get("test_mae_norm")))
    met2.metric("Test RMSE (norm)", _fmt_metric(tft_summary.get("test_rmse_norm")))
    met3.metric("Test R2", f"{tft_summary.get('test_r2', float('nan')):.4f}")
    st.write("### TFT benchmark modes")
    st.dataframe(tft_modes_df, use_container_width=True)

    if tft_pred_df is not None and not tft_pred_df.empty:
        st.write("### TFT predictions preview")
        st.dataframe(tft_pred_df.head(100), use_container_width=True)


def render_lstm_results(
    lstm_hist_df: pd.DataFrame | None,
    lstm_pred_df: pd.DataFrame | None,
    lstm_summary: dict | None = None,
) -> None:
    """Hiển thị kết quả LSTM — cấu trúc giống render_mamba_results."""
    if lstm_summary is not None:
        st.success("Train/Test LSTM hoàn tất")
        met1, met2, met3, met4 = st.columns(4)
        met1.metric("Val MAE (norm)",  _fmt_metric(lstm_summary.get("val_mae_norm")))
        met2.metric("Val RMSE (norm)", _fmt_metric(lstm_summary.get("val_rmse_norm")))
        met3.metric("Val R²",   f"{lstm_summary.get('val_r2',   float('nan')):.4f}")
        met4.metric("Locations", f"{int(lstm_summary.get('future_locations', 0)):,}")

        st.write("### Thống kê thời gian LSTM")
        st.write(
            {k: round(float(v), 2) if isinstance(v, float) else v
             for k, v in {
                 "train_only_sec": lstm_summary.get("train_only_sec", float("nan")),
                 "eval_sec":       lstm_summary.get("eval_sec",       float("nan")),
                 "forecast_sec":   lstm_summary.get("forecast_sec",   float("nan")),
                 "io_sec":         lstm_summary.get("io_sec",         float("nan")),
                 "run_sec":        lstm_summary.get("run_sec",        float("nan")),
                 "device":         lstm_summary.get("device",         "?"),
                 "split_train":    lstm_summary.get("split_train",    "?"),
                 "split_val":      lstm_summary.get("split_val",      "?"),
                 "split_test":     lstm_summary.get("split_test",     "?"),
             }.items()}
        )

    if lstm_hist_df is not None:
        st.write("### Lịch sử train LSTM")
        st.dataframe(lstm_hist_df, use_container_width=True)

    if lstm_pred_df is not None and not lstm_pred_df.empty:
        st.write("### Dự báo 24 giờ tiếp theo (LSTM)")
        st.dataframe(lstm_pred_df.head(300), use_container_width=True)



def render_comparison_table(
    summary: dict | None,
    tft_summary: dict | None,
    lstm_summary: dict | None,
) -> None:
    """Hiển thị bảng so sánh các model đã chạy (bỏ qua model None)."""
    rows = []

    if summary is not None:
        rows.append({
            "model": "Mamba",
            "test_mae_norm":  summary.get("test_mae_norm",  np.nan),
            "test_rmse_norm": summary.get("test_rmse_norm", np.nan),
            "test_r2":   summary.get("test_r2",        np.nan),
            "train_sec": summary.get("train_only_sec", np.nan),
            "run_sec":   summary.get("run_sec",        np.nan),
        })

    if tft_summary is not None:
        rows.append({
            "model": "TFT",
            "test_mae_norm":  tft_summary.get("test_mae_norm",  np.nan),
            "test_rmse_norm": tft_summary.get("test_rmse_norm", np.nan),
            "test_r2":   tft_summary.get("test_r2",        np.nan),
            "train_sec": tft_summary.get("train_only_sec", np.nan),
            "run_sec":   tft_summary.get("run_sec",        np.nan),
        })

    if lstm_summary is not None:
        rows.append({
            "model": "LSTM",
            "test_mae_norm":  lstm_summary.get("test_mae_norm",  np.nan),
            "test_rmse_norm": lstm_summary.get("test_rmse_norm", np.nan),
            "test_r2":   lstm_summary.get("test_r2",        np.nan),
            "train_sec": lstm_summary.get("train_only_sec", np.nan),
            "run_sec":   lstm_summary.get("run_sec",        np.nan),
        })

    if len(rows) < 1:
        return  # chưa có kết quả nào để so sánh

    st.divider()
    st.write(f"### 📊 So sánh {len(rows)} mô hình")
    st.caption("Đánh giá trên test set — cùng seed, cùng tập locations.")

    cmp_df = pd.DataFrame(rows)
    st.dataframe(cmp_df, use_container_width=True)

    if len(rows) >= 2:
        mae_best  = cmp_df.loc[cmp_df["test_mae_norm"].idxmin(),  "model"]
        rmse_best = cmp_df.loc[cmp_df["test_rmse_norm"].idxmin(), "model"]
        r2_best   = cmp_df.loc[cmp_df["test_r2"].idxmax(),   "model"]
        col1, col2, col3 = st.columns(3)
        col1.metric("Best MAE (norm)",  mae_best)
        col2.metric("Best RMSE (norm)", rmse_best)
        col3.metric("Best R²",   r2_best)


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
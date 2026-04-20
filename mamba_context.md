# Ngữ cảnh và Cách Hoạt Động của Mô hình Mamba (Mamba Time-Series)

Mô hình Mamba trong dự án của bạn là một giải pháp tiên tiến dùng để dự báo chất lượng không khí (AQI hoặc biến môi trường khác), thay thế cho các cơ chế Attention nặng nề trong Transformer. Mamba tập trung vào khả năng nắm bắt độ phụ thuộc chuỗi dài hiệu quả với chi phí bậc tuyến tính $O(N)$.

## 1. Cách Hoạt Động (Architecture)
- Vị trí: `scripts/train_mamba_aqi.py` (Class `TimeSeriesMambaRegressor`).
- Cơ chế: Mamba được xây dựng dựa trên **Selective State Space Models (SSMs)**. Khi đầu vào chuỗi thời gian đi vào mô hình, bộ vi phân Không gian trạng thái sẽ "xem qua" chuỗi đó và chọn lọc trạng thái giữ lại (Selective Scan) không thông qua phép tính ma trận Attention tốn kém.
- Embedding: Sử dụng `Location Token Embedding` để phân biệt cấu trúc/đặc tính chuỗi thời gian đối với nhiều thành phố (location_key). Từ đó ghép thêm 1 token không gian bên cạnh dữ liệu không khí.
- Lọc Đặc Trưng: Qua Linear Projection Projection và mạng đa tầng Mamba Block (`mamba_ssm.Mamba`), đầu ra được nén lại ở bước cuối của LayerNorm để đi qua các lớp Fully-Connected để thu về giá trị 1 chiều duy nhất (`horizon = 1`).

## 2. Cách Train (Huấn luyện)
- **Tạo mẫu (Windowing)**: Lấy một cửa sổ trượt (sliding window) kích thước `window_size` (vd: 24h) của toàn bộ các đặc trưng biến thiên (`X_Seq`) nhằm làm input, và nhãn `y` là AQI sau khoảng đó `horizon` tiếng.
- **Tối ưu hóa**: Dùng bộ tối ưuizer **AdamW** (kèm Weight decay) để điều chỉnh trọng số với hàm tính Loss (tùy chọn) là **HuberLoss** (chống outliers và bão hoà tốt) hoặc MSELoss.
- **Kỹ thuật phần cứng**: Quá trình lặp Train được trợ lực bởi kĩ thuật **AMP (Automatic Mixed Precision)** (Tính toán dấu phẩy động 16-bit nhằm tăng tốc trên GPU), và gộp vi phân (**Gradient Accumulation**) để tối ưu bộ nhớ VRAM đối với batch size thực tế lớn.
- **Quy trình Validation**: Dữ liệu chia mốc timeline `split_data_by_timeline()` thành Train 70%, Val 10%, Test 20%. Nếu epoch có val_loss giảm kỷ lục thì mô hình `best_mamba_aqi.pt` được lưu.

## 3. Cách Forecast (Dự báo)
Mamba trong thiết kế của bạn hoạt động như một loại **Direct Forecasting Model**. Chức năng forecast được lồng ở hàm `evaluate`.
- **Đầu vào (Input Window)**: Là 1 batch dữ liệu quá khứ kích thước `window_size`.
- **Dự Phóng**: Feed qua model một lần duy nhất để nhảy thẳng đến giá trị tại `T + horizon`. Khác với LSTM hiện tại của bạn lấy output gắn vào làm input dự đoán chuỗi tự động, Mamba "bắn thẳng" đến mốc cần dự đoán và đem Inverse Transform thông qua $mean$ và $std$ đã Scale ra kết quả.

## 4. Tập Dữ Liệu
Mamba sử dụng trực tiếp tệp Pandas DataFrame tại `dataset/2025.csv` (có thể linh hoạt tuỳ chỉnh tham số --data-path). Pipeline sẽ tự động:
- Parse ngày tháng `ts_utc`, dọn rác missing (`dropna`/`fillna`).
- Trích xuất các đặc trưng là số học (`numeric features`), mã hoá string `location_key`.
- Chuẩn hoá toàn thể dữ liệu đầu vào.

# LSTM Time-Series Forecasting - Project Context

## 1. Tổng quan dự án
Dự án này là một hệ thống dự báo chuỗi thời gian (time-series forecasting) với dữ liệu thực tế đa biến (multivariate) sử dụng mạng học sâu LSTM (Long Short-Term Memory) trên nền tảng PyTorch. 
Dự án được tối ưu hóa cho dữ liệu đo lường tại nhiều trạm/địa điểm (`location_key`) khác nhau. Quá trình triển khai bao gồm: đọc dữ liệu `.csv`, chuẩn hóa giá trị động (scale features) tách rời biến input và target, tạo block dữ liệu cửa sổ trượt (sliding windows) riêng rẽ cho từng địa điểm, huấn luyện và đánh giá chặt chẽ trên tập Test, tự động rút trích ra tệp dự báo 24 giờ trọn vẹn của ngày tiếp theo (Tomorrow).

## 2. Cấu trúc thư mục và Vai trò của từng file

Dưới đây là cấu trúc toàn bộ cây thư mục đang có của dự án và giải thích chi tiết cho từng file/thư mục:

```text
LSTM-Time-Series-Forecasting/
├── README.md
├── LICENSE
├── requirements.txt
├── context.md
├── data/
│   └── 2025.csv
├── src/
│   ├── evaluate.py
│   ├── train_lstm.py
│   └── utils.py
└── outputs/
    └── [location_key]/
        ├── metrics.csv
        ├── future_forecast.csv
        └── best_lstm.pt
```

- **`README.md` & `requirements.txt`**: Cung cấp thông tin dự án và danh sách các thư viện hỗ trợ (`torch`, `numpy`, `pandas`, `scikit-learn`,...).
- **`context.md`**: Tài liệu lưu trữ cấu trúc tổng quan và nguyên lý hiện hành của dự án này.

### 2.1 Thư mục `data/`
- **Các file CSV (VD: `2025.csv`)**: Kho chứa dữ liệu lưu trữ chuỗi thời gian đã được dọn dẹp, phân cấp rạch ròi theo từng trạm đo (`location_key`) và mốc thời gian (`ts_utc`), bao gồm biến mục tiêu và các biến đặc trưng (features) quan trắc.

### 2.2 Thư mục `src/`
Chứa mã nguồn lõi cho các quá trình Pipeline:
- **`utils.py`**: Module dùng chung tiện lợi:
  - Hàm `make_windows_grouped(...)`: Sinh ra sliding window 3D (`Batch`, `Time`, `Features`) nghiêm ngặt và liền mạch, ngắt quãng an toàn (không bị rò rỉ hay chồng chéo mảng thời gian giữa các `location_key` phân rẽ).
  - Hàm `scale_features(...)`: Tách rời phương thức của `X_scaler` và `y_scaler` để bảo đảm chuẩn hóa mục tiêu dựa đúng gốc tọa độ tự nhiên. Trả ngược số liệu hoàn hảo.
  - Bộ tính toán KPI: `rmse`, `mae`, `mape`.
- **`train_lstm.py`**: Trái tim huấn luyện AI LSTM.
  - Tự động lập chỉ mục, định danh cột (Numeric) và chọn lọc đối tượng qua tham số truyền `--location`.
  - Phân tách tập dữ liệu thành ba nhóm Train (70%) / Val (10%) / Test (20%).
  - Xây dựng mô hình Mạng nơ-ron `LSTMForecaster` với chuẩn `batch_first=True`.
  - Kết thúc vòng loop: tiến hành đánh giá sai số trên 20% lượng mẫu của tập Test vào CSV, và lưu mô hình `best_lstm.pt` - nhúng gọn (bundle) linh hoạt cả hai scaler vào cùng một file.
  - Khâu hậu xử lý: Chạy phóng timeline tầm xa và tự động tinh chuẩn trích xuất đúng 24h quy lai thành "phần của ngày mai" vào tệp dự báo. Loại bỏ biểu đồ để tối đa tính gọn gàng hiệu năng lệnh CLI.
- **`evaluate.py`**: Đoạn kịch bản chuyên dụng khi chỉ có nhu cầu "Rút file mô hình và chấm điểm" hoặc "Dự phóng tập mới" nếu không muốn phí thời gian Train loop lại từ đầu. Sử dụng chung bộ thư mục đầu ra giống hàm train.

### 2.3 Thư mục `outputs/`
Thư mục gốc chứa thành quả. Mỗi khi chạy mô hình sinh ra trạm nào đó (ví dụ: `angiang_longxuyen`), mọi rác thải sẽ triệt tiêu và kết quả cô đọng được phân cấp đẩy vào bên trong thư mục con tương ứng nhằm tránh nhiễu:
- **`best_lstm.pt`**: File lưu các ma trận Parameter & Weights xịn nhất tự động bắt bằng EarlyStopping. Trọng số này gắn chặt kèm với `X_scaler` và `y_scaler` để di dời portable hoàn hảo.
- **`metrics.csv`**: File đo lường chỉ số độ lỗi cho Test-Set (RMSE, MAE, MAPE).
- **`future_forecast.csv`**: Tập DataFrame chứa mã trạm `location_key` vừa chèn, trục thời gian `ts_utc`, và giá trị biến thiên nguyên thủy mà mô hình đã vẽ nên cho mục tiêu duy nhất: 24 tiếng sau của ngày mới nhất.

## 3. Luồng làm việc (Workflow)

1. **Gom Dữ liệu**: Thả các tệp tin lưu lượng CSV đảm bảo chứa bộ đôi Index (`ts_utc` và `location_key`) vào folder `data/`.
2. **Train Trạm/Khu Vực Bất Kỳ**: Gọi đoạn script `src/train_lstm.py` tại terminal. 
   - *Lệnh Khuyến nghị (Để lấy trọn thời gian hẫng đến hết ngày mai)*:
     `python src/train_lstm.py --target aqi --location angiang_longxuyen --horizon 48 --epochs 5`
3. **Thu hoạch**: Vào `outputs/[tên_vị_trí]/` để bắt trọn các files thành phẩm của khu vực ấy.

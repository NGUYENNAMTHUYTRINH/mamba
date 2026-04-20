# Ngữ cảnh và Cách Hoạt Động của Mô hình LSTM

Mô hình LSTM (Long Short-Term Memory) trong cấu trúc dự án của bạn đóng vai trò là giải pháp truyền thống (baseline vững chắc) cho bài toán Dữ liệu chuỗi thời gian nhiều biến. Đây là thuật toán Recurrent Neural Network (RNN) chuyên biệt xử lý dữ liệu tuần tự từ dĩ vãng đến hiện tại.

## 1. Cách Hoạt Động (Architecture)
- Vị trí: `LSTM-Time-Series-Forecasting/src/train_lstm.py` (Class `LSTMForecaster`).
- Lõi Mạng (Core Block): Dành cho việc lưu trữ các trạng thái Cell State dài hạn và Hidden State ngắn hạn, LSTM sử dụng 3 cánh cổng toán học (Input, Forget, Output Gate) để học các mẫu dữ liệu xu hướng khí hậu thời gian. Trong source code, bạn đang tận dụng lớp mạng `nn.LSTM` gộp sẵn trong PyTorch.
- Metadata Location (Chỉ mục địa lý): Khác so với bản gốc của RNN chung chung, bài toán này bạn trang bị cơ chế dùng `nn.Embedding` (8 dimensions) khi có nhiều hơn một địa phương huấn luyện. Trọng số của thành phố được nối ghép với thuộc tính đầu vào để phân biệt rõ đặc thù khí hậu từng tỉnh.
- Output Layer: Giá trị cuối chuỗi của nấc Hidden state (`h_last`) đi vào mạng tuyến tính cơ bản (`nn.Linear`) ra kết quả mục tiêu của 1 unit thời gian (thường là horizon=1).

## 2. Cách Train (Huấn luyện)
- **Tiền xử lý chuỗi (Windowing)**: Code tạo các cặp dữ liệu `X`, `y` qua hàm chia `make_windows_grouped` (với thông số window size là `lookback`). 
- **Thiết lập tối ưu optimizer**: Sử dụng hàm hao phí dạng MSELoss() dùng để phạt trọng lượng chênh lệch bình phương kết hợp **Adam Optimizer**.
- **Epoch Training**: Chia tách làm 3 tập dữ liệu `Train(70%), Validation(10%), Test(20%)`. Training diễn ra qua các vòng lặp (Epoch), tại mỗi epoch tự động tính các chỉ số RMSE, MAE, R2.
- **Tiêu chí kết thúc sớm (Early Stopping)**: Nếu sau 5 Epoch mà Val-Loss không có hiện tượng giảm đi so với kỷ lục tốt nhất tiếp theo `(stale >= 5)`, quy trình Train sẽ tự động được ngắt để tránh Overfitting. Mô hình ưu tú nhất ghi vào đĩa: `best_lstm.pt`.

## 3. Cách Forecast (Dự báo Tương Lai Thực Sự)
Trong thiết kế project này, LSTM được setup thiên hướng mạnh theo kịch bản: **Auto-Regressive Forecasting** (Dự đoán cuốn chiếu đa bước).
- Nghĩa là, dùng 1 cánh cửa `lookback` trong quá khứ chứa đủ các tính năng thực thụ, nó suy luận ra tiếng thứ nhất (Tiếng: 1).
- Tiếp theo, vì ở tương lai thực sự chúng ta sẽ *không có* chỉ số môi trường, vòng lặp tạo 1 hàng dữ liệu bản sao, thay giá trị AQI thành giá trị `Vừa Dự Đoán`. Nhét hàng đó nhích sang thay dần vị trí cuối cửa sổ, cắt bỏ rác cũ đầu cửa sổ (Sliding window logic).
- Tiếp tục dự phóng (inference) mô hình 1 lần nữa ở cửa sổ lai tạo này sẽ ra (Tiếng: 2).
- Vòng lặp tuân theo biến `forecast_steps` (mặc định ra thêm 24 giờ). Kết quả gỡ Scaling (inverse) về dạng số thực AQI của con người và lưu file `future_forecast.csv` kèm dãy Timestamp ảo (`ts_utc` sinh ra tự động dạng tần suất pd.Timdelta = 1h).

## 4. Tập Dữ Liệu Được Gọi Từ Đâu
Nguồn cấp cho LSTM dựa trên một tệp gốc tổng ở tham số argument parser `--data-path` (mặc định lấy đúng `dataset/2025.csv`). Cấu trúc xử lý cũng parse thời gian gỡ Timezone sang dạng UTC trơn và chỉ lấy các thành phần biến thiên dạng `int` hoặc `float` tạo thành `numeric features`. Toàn bộ dữ liệu được chuẩn hóa tỉ lệ bằng MinMaxScaler hoặc kỹ thuật scale tương tự (`scale_features()`).

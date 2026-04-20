# Ngữ cảnh và Cách Hoạt Động của Mô hình Transformer / TFT (Temporal Fusion Transformer)

Mô hình Transformer đóng vai trò khá "đồ sộ" trong dự án vì cách xây dựng ở thư mục `Transformer_Timeseries`. Bạn đang dùng cấu trúc hỗ trợ cả 3 phiên bản: Baseline Transformer, GRN-Transformer, và Temporal Fusion Transformer (TFT). TFT đặc biệt vô cùng mạnh nếu phải đối phó với dữ liệu nhiều đặc trưng lộn xộn (ngày tháng, số thực, dữ liệu tĩnh).

## 1. Cách Hoạt Động (Architecture)
- **Vị trí**: Quản lý ở thư mục `Transformer_Timeseries/models` (Class `tft_model.TFT` hoặc `Transformer`). Điều khiển dòng chảy tại `trainer.py` kết nối YAML (`conf/air_quality.yaml`).
- **Mã hoá theo loại biến**: Đặc thù lớn nhất của TFT là chia chẻ dữ liệu đầu vào thành 3 dạng riêng biệt để đi qua Lớp mã hoá Input:
   1. Static Variables (Biến tĩnh): Ví dụ như địa điểm (location_key). Không đổi theo thời gian.
   2. Known Future Variables (Biến đã biết ở tương lai): VD như Ngày Trong Mùa, Giờ Trong Ngày.
   3. Unknown Past Variables (Biến quá khứ chưa biết tương lai): Ví dụ: Độ ẩm, Lượng Mưa, Thông số độc hại của AQI hôm nay.
- **Mơ-đun cốt lõi**:
  - `Gated Residual Network (GRN)`: Bộ lọc thông minh quyết định đặc trưng phụ nào (ví dụ độ che phủ mây) tại thời điểm T có quan trọng thực sự để đoán AQI không. Không thì bỏ qua linh hoạt.
  - `Multi-Head Attention (Interpretable)`: Tính toán mức tương quan giữa khung thời gian (past timesteps) với cột mốc cần dự đoán tương lai để tìm hiểu chu kỳ lịch sử. 
- Đầu ra của TFT có thể là mô phỏng theo **Quantile Forecast (Dự báo khoảng/ xác suất)** thay vì giá trị cố định, VD tính p10 (Lạc quan), p50 (Trung bình), p90 (Khốc liệt nhất). Nếu chọn biến thể config PointForecast, kiến trúc lại thay Output Layer để thành một số cụ thể.

## 2. Cách Train (Huấn luyện)
- **Tạo Dataset đặc thù**: Không load chuỗi cứng nhắc mà sử dụng kiến trúc Class Format (`TSDataset`) từ `data_formatters/air_quality.py`. Dữ liệu sẽ chia các thuộc tính `id_col`, `time_col`, `target_col`, `input_cols`.
- **Tối Ưu Hoá Loss**: 
   - *Quantile Loss*: Phạt mạnh việc chênh lệch phân phối (nếu set config muốn tính tỷ lệ rủi ro bằng dự báo đường bao 10-50-90 percentiles). 
   - *Huber/MSE Loss*: Nếu config mô hình bắt chước trả về Point Forecasting giống Mamba/LSTM.
   - Thường chọn **Adam / AdamW optimizer**. Cùng kết hợp ReduceLROnPlateau Scheduler - tự động hạ thấp Learning rate khi việc học bị đứng trên Val Set. Tính năng AMP và Gradient Accumulation cũng được áp dụng hỗ trợ tại file `trainer.py`.

## 3. Cách Forecast / Đánh Giá (Dự báo)
Trong module Transformer, phương pháp đánh giá (Test) khác biệt phức tạp hơn LSTM đôi chút do thiết kế kiến trúc theo luồng chung Encoder-Decoder:
- Các bước tương lai ở Test dataset (mốc `encoder_steps` trở đi) ban đầu sẽ bị biến thành giá trị Dummy (bị che lấp 0 hoặc 1) để tránh Model học trộm.
- Sau đó sử dụng quá trình **Autoregressive Loop**: với mô hình chuỗi thì dự đoán ra 1 điểm `point_step` tại chiều dài `pred_len`. Gắn giá trị điểm vừa được dự báo đó để làm thành nền tảng đệm cho bước `encoder_steps + 1`. Lặp kín vòng và gỡ màng scale (inverse target).

## 4. Tập Dữ Liệu Gọi Nguồn
TFT không fix chết nguồn CSV bằng Args. Hệ thống Transformer đọc File cấu hình tên YAML (Ví dụ: `conf/air_quality.yaml`). Trong tham số `ds_name: 'air_quality'` có lẽ sẽ trỏ formatter để kết nối vào Dataset mặc định (cũng từ `dataset/2025.csv`). Format Pandas sẽ convert các thuộc tính categorical index (`CategoricalDtype`) cũng như thực hiện scale cục bộ từng column tương xứng theo cách cấu hình cho `AirQualityFormatter`.

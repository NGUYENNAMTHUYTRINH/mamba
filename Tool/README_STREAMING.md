# Air Quality Streaming Service

Đây là service streaming tự động thu thập dữ liệu chất lượng không khí từ API WAQI và lưu vào MinIO + PostgreSQL.

## 🚀 Cách Deploy

### Option 1: Docker Compose (Khuyến nghị)

```bash
# Build và deploy cùng với toàn bộ stack
docker-compose up -d --build air-quality-streaming

# Hoặc sử dụng script tiện ích (Windows)
streaming_deploy.bat deploy

# Hoặc sử dụng script tiện ích (Linux/Mac)
chmod +x streaming_deploy.sh
./streaming_deploy.sh deploy
```

### Option 2: Docker Standalone

```bash
# Build image
docker build -t air-quality-streaming:latest .

# Run container
docker run -d --name air-quality-streaming \
    --restart unless-stopped \
    --network airflow-docker_default \
    -e POSTGRES_HOST=postgres \
    -e POSTGRES_DB=airflow \
    -e POSTGRES_USER=airflow \
    -e POSTGRES_PASSWORD=airflow \
    -e MINIO_HOST=minio:9000 \
    -e MINIO_ACCESS_KEY=admin \
    -e MINIO_SECRET_KEY=admin123 \
    -e WAQI_TOKEN=YOUR_TOKEN \
    -e STREAMING_INTERVAL=300 \
    -v $(pwd)/logs:/app/logs \
    air-quality-streaming:latest
```

## ⚙️ Cấu hình Environment Variables

| Variable | Mô tả | Mặc định |
|----------|--------|----------|
| `POSTGRES_HOST` | PostgreSQL hostname | `postgres` |
| `POSTGRES_DB` | Database name | `airflow` |
| `POSTGRES_USER` | Database user | `airflow` |
| `POSTGRES_PASSWORD` | Database password | `airflow` |
| `MINIO_HOST` | MinIO hostname:port | `minio:9000` |
| `MINIO_ACCESS_KEY` | MinIO access key | `admin` |
| `MINIO_SECRET_KEY` | MinIO secret key | `admin123` |
| `WAQI_TOKEN` | WAQI API token | *required* |
| `STREAMING_INTERVAL` | Interval giữa các lần fetch (giây) | `300` (5 phút) |

## 📊 Monitoring

### Xem logs realtime
```bash
docker logs -f air-quality-streaming
```

### Kiểm tra status
```bash
# Sử dụng script tiện ích
streaming_deploy.bat status

# Hoặc Docker command
docker ps | grep air-quality-streaming
docker stats air-quality-streaming
```

### Restart service
```bash
streaming_deploy.bat restart
```

## 🔧 Quản lý với Airflow

Service có tích hợp DAG `streaming_management_dag` để:

- ✅ **Health check** mỗi giờ
- 📊 **Monitor** database và MinIO
- 🔄 **Auto restart** nếu service fail
- 📋 **Generate reports** về tình trạng hệ thống
- ⚡ **Manual trigger** để thu thập dữ liệu ngay lập tức

## 📂 Cấu trúc dữ liệu

### MinIO (Data Lake)
```
air-quality/
├── waqi_raw/
│   ├── city=Hanoi_Hoan_Kiem/
│   │   └── year=2025/month=09/day=22/
│   │       └── 2025-09-22T10:30:00Z.json
│   └── city=Ho_Chi_Minh_City_District_1/
└── historical/
    └── station=Hanoi_Hoan_Kiem/
```

### PostgreSQL (Data Warehouse)
- `dim_air_quality_stations` - Thông tin các trạm
- `fact_air_quality_measurements` - Dữ liệu đo lường

## 🚨 Troubleshooting

### Container không start
```bash
# Xem logs lỗi
docker logs air-quality-streaming

# Kiểm tra kết nối
docker exec -it air-quality-streaming python -c "
import psycopg2
conn = psycopg2.connect(host='postgres', user='airflow', password='airflow', dbname='airflow')
print('DB OK')
"
```

### Dữ liệu không được thu thập
1. Kiểm tra WAQI token có đúng không
2. Kiểm tra kết nối network giữa containers
3. Xem logs để debug API errors

### Performance issues
1. Tăng `STREAMING_INTERVAL` nếu API rate limit
2. Monitor MinIO disk space
3. Kiểm tra PostgreSQL connection pool

## 📈 Thống kê

- **~70 trạm** đo lường trên toàn Vietnam  
- **5 phút/lần** thu thập dữ liệu
- **~20,000 measurements/ngày** 
- **Auto restart** khi có lỗi
- **Retention logs** 3 files x 10MB

## 🔒 Bảo mật

- Container chạy với non-root user
- Health checks tích hợp
- Auto restart `unless-stopped`
- Logs rotation để tránh disk full
- Network isolation với Docker

## 📞 Hỗ trợ

Nếu có vấn đề, kiểm tra:
1. `streaming_deploy.bat status` - Tình trạng service
2. `docker logs air-quality-streaming` - Logs chi tiết  
3. Airflow UI > DAGs > `streaming_management_dag` - Monitoring dashboard
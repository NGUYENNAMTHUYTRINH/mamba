# 🧪 LỆNH TEST DỮ LIỆU STREAMING SERVICE

## 1️⃣ **KIỂM TRA CONTAINER STATUS**
```powershell
# Container đang chạy
docker ps | findstr air-quality

# Health check
docker inspect air-quality-streaming --format='{{.State.Health.Status}}'
```

## 2️⃣ **XEM LOGS REALTIME**
```powershell
# Log filtering (PowerShell)
docker logs air-quality-streaming | Select-String "completed|processed|ERROR|WARNING" | Select-Object -Last 10

# Log theo thời gian thật
docker exec air-quality-streaming tail -f /app/logs/streaming.log

# Chỉ xem streaming rounds
docker logs air-quality-streaming | findstr "Streaming round completed"
```

## 3️⃣ **TEST POSTGRESQL DATA**
```powershell
# Tổng số measurements
docker exec airflow-docker-postgres-1 psql -U airflow -d airflow -c "SELECT COUNT(*) FROM fact_air_quality_measurements;"

# Số lượng stations
docker exec airflow-docker-postgres-1 psql -U airflow -d airflow -c "SELECT COUNT(*) FROM dim_air_quality_stations;"

# 10 measurements mới nhất
docker exec airflow-docker-postgres-1 psql -U airflow -d airflow -c "SELECT s.station_name, f.aqi, f.pm25, f.measured_at FROM fact_air_quality_measurements f JOIN dim_air_quality_stations s ON f.station_id = s.station_id WHERE f.measured_at IS NOT NULL ORDER BY f.measured_at DESC LIMIT 10;"

# Stations có data trong 1 giờ gần nhất
docker exec airflow-docker-postgres-1 psql -U airflow -d airflow -c "SELECT COUNT(DISTINCT s.station_name) as active_stations FROM fact_air_quality_measurements f JOIN dim_air_quality_stations s ON f.station_id = s.station_id WHERE f.created_at >= NOW() - INTERVAL '1 hour';"
```

## 4️⃣ **TEST MINIO DATA** (nếu có mc client)
```powershell
# Xem buckets
docker exec airflow-docker-minio-1 mc alias set local http://localhost:9000 minio minio123
docker exec airflow-docker-minio-1 mc ls local/

# Count raw JSON files (nếu bucket tồn tại)
docker exec airflow-docker-minio-1 mc find local/waqi-bucket/ --name "*.json" | wc -l
```

## 5️⃣ **HEALTH CHECK API** (nếu có)
```powershell
# Test container network
docker exec air-quality-streaming curl -s http://localhost:8080/health || echo "No health endpoint"
```

## 6️⃣ **QUICK STATUS SUMMARY**
```powershell
echo "=== STREAMING SERVICE STATUS ==="
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | findstr air-quality
echo ""
echo "=== DATA COUNTS ==="
docker exec airflow-docker-postgres-1 psql -U airflow -d airflow -c "SELECT 'Measurements' as table_name, COUNT(*) as count FROM fact_air_quality_measurements UNION SELECT 'Stations', COUNT(*) FROM dim_air_quality_stations;"
echo ""  
echo "=== LATEST ACTIVITY ==="
docker logs air-quality-streaming | Select-String "Streaming round completed" | Select-Object -Last 3
```

## 🎯 **KẾT QUẢ MONG MUỐN:**
- Container: ✅ healthy/running
- PostgreSQL: ✅ 40,000+ measurements, 80+ stations  
- Logs: ✅ "Streaming round completed: 28 stations processed"
- MinIO: ✅ Raw JSON files (nếu cấu hình đúng)

## 🚨 **TROUBLESHOOTING:**
```powershell
# Restart service nếu cần
docker restart air-quality-streaming

# Check container resources
docker stats air-quality-streaming --no-stream

# Network connectivity test
docker exec air-quality-streaming ping -c 3 postgres
docker exec air-quality-streaming ping -c 3 minio
```
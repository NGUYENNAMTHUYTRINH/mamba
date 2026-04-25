This project now includes a MinIO service for storing raw Air Quality JSON objects and mounts `requirements.txt` into Airflow containers so additional Python dependencies are installed at startup.

Quick steps (PowerShell) to recreate the stack and ensure Airflow installs the requirements:

# 1. Stop running services (if any)
docker compose -f docker-compose.yaml down

# 2. Remove stale volumes for Airflow images if you want a fresh start (optional)
# WARNING: This will delete Postgres, Kafka, MinIO data in volumes
# docker volume rm airflow-docker_postgres-db-volume airflow-docker_kafka-data airflow-docker_minio-data

# 3. Start the stack (this will cause Airflow containers to pip install the contents of /opt/airflow/requirements.txt)
docker compose -f docker-compose.yaml up -d --build

# 4. Tail logs for the scheduler to see dependency installation and health
# (replace with other service names as needed)
docker compose -f docker-compose.yaml logs -f airflow-scheduler

Notes:
- Installing requirements at container startup is convenient for development but slow; for production build a custom image instead.
- MinIO is exposed on host port 9004 (http://localhost:9004). Use credentials: admin / admin123.
- Ensure no other service is using ports 9004, 9092, 9093, 2181, or 8081.

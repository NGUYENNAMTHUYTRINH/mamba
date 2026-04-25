# aqi_bronze_incremental.py
from datetime import datetime
from airflow import DAG
from airflow.operators.bash import BashOperator

DEFAULT_LOC_PATH = "/opt/EDA_AQI/Jobs/src/eda/locations.jsonl"

with DAG(
    dag_id="aqi_bronze_incremental",
    start_date=datetime(2025, 11, 1),
    schedule="0 * * * *",   # 5 phút 1 lần: chỉ để ingest incremental vào Bronze
    catchup=False,
    max_active_runs=1,
    tags=["aqi", "bronze", "incremental"],
) as dag:

    incr_cmd = r"""
    set -e
    LOC_PATH="{{ dag_run.conf.get('locations', '""" + DEFAULT_LOC_PATH + r"""') }}"
    echo "Using locations file: $LOC_PATH"

    python3 /opt/EDA_AQI/Jobs/src/eda/load_raw_data/bronze_raw.py \
      --mode incremental \
      --locations "$LOC_PATH"
    """

    bronze_incremental = BashOperator(
        task_id="bronze_incremental",
        bash_command=incr_cmd,
        env={
            "MINIO_HOST": "{{ var.value.get('MINIO_HOST', 'minio:9000') }}",
            "MINIO_ACCESS_KEY": "{{ var.value.get('MINIO_ACCESS_KEY', 'admin') }}",
            "MINIO_SECRET_KEY": "{{ var.value.get('MINIO_SECRET_KEY', 'admin123') }}",
            "MINIO_BUCKET": "{{ var.value.get('MINIO_BUCKET', 'air-quality') }}",
            "MINIO_SECURE": "{{ var.value.get('MINIO_SECURE', 'false') }}",
        },
    )

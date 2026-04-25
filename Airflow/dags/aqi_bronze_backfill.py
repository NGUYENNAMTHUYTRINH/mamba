# aqi_bronze_backfill.py
from datetime import datetime
from airflow import DAG
from airflow.operators.bash import BashOperator

DEFAULT_LOC_PATH = "/opt/EDA_AQI/Jobs/src/eda/locations.jsonl"

with DAG(
    dag_id="aqi_bronze_backfill",
    start_date=datetime(2025, 11, 1),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=["aqi", "bronze", "backfill"],
) as dag:

    cmd = r"""
    set -e
    echo "PWD: $(pwd)"
    echo "List /opt/EDA_AQI/Jobs/src/eda:"
    ls -lah /opt/EDA_AQI/Jobs/src/eda || true

    # chọn locations: nếu dag_run.conf có thì xài, không thì xài default
    LOC_PATH="{{ dag_run.conf.get('locations', '""" + DEFAULT_LOC_PATH + r"""') }}"
    echo "Using locations file: $LOC_PATH"

    python3 /opt/EDA_AQI/Jobs/src/eda/load_raw_data/bronze_raw.py \
      --mode backfill \
      --start-date {{ dag_run.conf.get('start_date', '2024-01-01') }} \
      --end-date {{ dag_run.conf.get('end_date', ds) }} \
      --chunk-days {{ dag_run.conf.get('chunk_days', 60) }} \
      --locations "$LOC_PATH"
    """

    run = BashOperator(
        task_id="bronze_backfill",
        bash_command=cmd,
        env={
            "MINIO_HOST": "{{ var.value.get('MINIO_HOST', 'minio:9000') }}",
            "MINIO_ACCESS_KEY": "{{ var.value.get('MINIO_ACCESS_KEY', 'admin') }}",
            "MINIO_SECRET_KEY": "{{ var.value.get('MINIO_SECRET_KEY', 'admin123') }}",
            "MINIO_BUCKET": "{{ var.value.get('MINIO_BUCKET', 'air-quality') }}",
            "MINIO_SECURE": "{{ var.value.get('MINIO_SECURE', 'false') }}",
        },
    )

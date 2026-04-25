# aqi_eda_daily.py
from datetime import datetime
from airflow import DAG
from airflow.operators.bash import BashOperator

with DAG(
    dag_id="aqi_eda_daily",
    start_date=datetime(2025, 11, 1),
    schedule="10 0 * * *",   # 00:10 mỗi ngày
    catchup=False,
    max_active_runs=1,
    tags=["aqi", "eda", "build_global", "cleaning"],
) as dag:

    build_cmd = r"""
    set -e
    YEAR="{{ ds[:4] }}"
    echo "▶ Build yearly global for year=$YEAR"
    python3 /opt/EDA_AQI/Jobs/src/eda/load_raw_data/bronze_build_global.py --year "$YEAR"
    """

    build_global = BashOperator(
        task_id="build_global_year",
        bash_command=build_cmd,
        env={
            "MINIO_HOST": "{{ var.value.get('MINIO_HOST', 'minio:9000') }}",
            "MINIO_ACCESS_KEY": "{{ var.value.get('MINIO_ACCESS_KEY', 'admin') }}",
            "MINIO_SECRET_KEY": "{{ var.value.get('MINIO_SECRET_KEY', 'admin123') }}",
            "MINIO_BUCKET": "{{ var.value.get('MINIO_BUCKET', 'air-quality') }}",
            "MINIO_SECURE": "{{ var.value.get('MINIO_SECURE', 'false') }}",
        },
    )

    clean_cmd = r"""
    set -e
    YEAR="{{ ds[:4] }}"
    echo "▶ Run EDA & Cleaning (Silver, chunked) for year=$YEAR"

    python3 /opt/EDA_AQI/Jobs/src/eda/pre-processing/data_preprocessing.py --year "$YEAR"
    """

    eda_clean_global = BashOperator(
        task_id="silver_clean_global",
        bash_command=clean_cmd,
        env={
            "MINIO_HOST": "{{ var.value.get('MINIO_HOST', 'minio:9000') }}",
            "MINIO_ACCESS_KEY": "{{ var.value.get('MINIO_ACCESS_KEY', 'admin') }}",
            "MINIO_SECRET_KEY": "{{ var.value.get('MINIO_SECRET_KEY', 'admin123') }}",
            "MINIO_BUCKET": "{{ var.value.get('MINIO_BUCKET', 'air-quality') }}",
            "MINIO_CLEAN_BUCKET": "{{ var.value.get('MINIO_CLEAN_BUCKET', 'air-quality-clean') }}",
            "MINIO_SECURE": "{{ var.value.get('MINIO_SECURE', 'false') }}",
        },
    )

    build_global >> eda_clean_global

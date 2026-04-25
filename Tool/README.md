Quick setup for running the Kafka producer script and installing Python dependencies (Windows PowerShell)

1) Create and activate a venv (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell prevents running the Activate script, run as Administrator and allow scripts:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

2) Install dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

3) Run the producer script that sends test messages to Kafka (expects Docker Compose stack running):

```powershell
python .\\dags\\produce_test.py
```

Notes
- The `produce_test.py` script uses `kafka:9092` as the bootstrap server because it's intended to run inside the Docker Compose network (e.g. in the `airflow-apiserver` container or `airflow-cli`). If you run the producer from the host, change `BOOTSTRAP` in the script to `localhost:9092`.
- To make the dependency persist inside Airflow containers you can either add `_PIP_ADDITIONAL_REQUIREMENTS: "kafka-python"` to `docker-compose.yaml` under Airflow services, or build a custom Airflow image that includes `kafka-python`.


MINIO_HOST=172.27.91.163:9000
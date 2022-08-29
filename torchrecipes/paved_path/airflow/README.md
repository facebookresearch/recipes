# Airflow example
1. Install and start an Airflow server
```bash
./setup.sh
```
> **_NOTE:_**: Airflow UI can be accessed at http://0.0.0.0:8080 (replace the address with your EC2 instance address for public access). Learn more about airflow from [Quick Start](https://airflow.apache.org/docs/apache-airflow/stable/start/local.html)

2. Create a dag config. See the example in `train_charnn.py`

3. Set `dag_folder` to folder containing the dag config in `~/airflow/airflow.cfg`. Such that Airflow can discover your dag configs.

3. Run a task instance
```bash
airflow tasks run train_charnn train 2022-08-01
```
> **_NOTE:_**: the instance can be monitored in UI: http://0.0.0.0:8080/taskinstance/list

4. Backfill the dag over 2 days
```bash
airflow dags backfill train_charnn --start-date 2022-08-01 --end-date 2022-08-02
```
> **_NOTE:_**: the dag runs can be monitored in UI: http://0.0.0.0:8080/dagrun/list/
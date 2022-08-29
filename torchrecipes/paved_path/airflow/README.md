# Airflow example
1. Install and start an airflow server
```batch
./setup.sh
```
Learn more about airflow from [Quick Start](https://airflow.apache.org/docs/apache-airflow/stable/start/local.html)

2. Create a dag
See the example in `train_charnn.py`

3. Run a task instance
```batch
airflow tasks run train_charnn train 2022-08-01
```

4. Backfill the dag over 2 days
```batch
airflow dags backfill train_charnn --start-date 2022-08-01 --end-date 2022-08-02
```
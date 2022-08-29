# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from datetime import datetime, timedelta

import boto3.session

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.hooks.batch_waiters import BatchWaitersHook

REGION = "us-west-2"
JOB_QUEUE = "torchx-gpu"
ECR_URL = os.environ["ECR_URL"]


default_args = {
    "depends_on_past": False,
    "email": ["hudeven@gmail.com"],
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 0,
    "retry_delay": timedelta(minutes=5),
}


dag = DAG(
    "train_charnn",
    default_args=default_args,
    description="A DAG to train charnn in AWS Batch",
    schedule_interval="@daily",
    catchup=False,
    start_date=datetime(2022, 8, 1),
    tags=["aws_batch"],
)


train = BashOperator(
    task_id="train",
    bash_command="""AWS_DEFAULT_REGION=$REGION \
        torchx run --workspace '' -s aws_batch \
        -cfg queue=$JOB_QUEUE,image_repo=$ECR_URL/charnn dist.ddp \
        --script charnn/main.py --image $ECR_URL/charnn:latest \
        --cpu 8 --gpu 2 -j 1x2 --memMB 20480 2>&1 \
        | grep -Eo aws_batch://torchx/$JOB_QUEUE:main-[a-z0-9]+""",
    env={
        "REGION": REGION,
        "JOB_QUEUE": JOB_QUEUE,
        "ECR_URL": ECR_URL,
    },
    dag=dag,
    do_xcom_push=True,
)


def wait_for_batch_job(**context) -> bool:
    session = boto3.session.Session()
    client = session.client("batch", region_name=REGION)
    job = context["ti"].xcom_pull(task_ids="train")
    job_desc = job.split("/")[-1]
    queue_name, job_name = job_desc.split(":")
    job_id = client.list_jobs(
        jobQueue=queue_name,
        filters=[{"name": "JOB_NAME", "values": [job_name]}],
    )["jobSummaryList"][0]["jobId"]
    waiter = BatchWaitersHook(region_name=REGION)
    try:
        waiter.wait_for_job(job_id)
        return True
    except Exception:
        return False


wait_for_job = PythonOperator(
    task_id="wait_for_job",
    python_callable=wait_for_batch_job,
    dag=dag,
)


parse_output = BashOperator(
    task_id="parse_output",
    bash_command="output: {{ ti.xcom_pull(task_ids='wait_for_job') }}",
    dag=dag,
)


train >> wait_for_job >> parse_output

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import uuid
from datetime import datetime, timedelta

import boto3.session

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.hooks.batch_waiters import BatchWaitersHook
from airflow.providers.amazon.aws.operators.batch import BatchOperator

# AWS Elastic Container Registry(ECR) Configs
REGION = "us-west-2"
ECR_URL = os.environ["ECR_URL"]
IMAGE = "613447952645.dkr.ecr.us-west-2.amazonaws.com/charnn:latest"

# AWS Batch configs
JOB_QUEUE_GPU = "torchx-gpu"
JOB_QUEUE_GENERAL = "general-ec2-cpu"
JOB_DEFINITION_GENERAL = "charnn-general-cpu"

work_dir = "s3://paved-path-prototype/charnn"
job_name = f"run-{uuid.uuid4()}"
module_path = os.path.join(work_dir, job_name, "modules/last.pt")
exported_module_path = os.path.join(work_dir, job_name, "modules/exported.pt")

logger = logging.getLogger("charnn_dag")

default_args = {
    "depends_on_past": False,
    "email": ["hudeven@gmail.com"],
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 0,
    "retry_delay": timedelta(minutes=5),
}


dag = DAG(
    "charnn_dag",
    default_args=default_args,
    description="A DAG to train charnn in AWS Batch",
    schedule_interval="@daily",
    catchup=False,
    start_date=datetime(2022, 8, 1),
    tags=["aws_batch"],
)


# This example uses torchx CLI with BashOperator.
# We can also use PythonOperator to achive it.
train = BashOperator(
    task_id="train",
    bash_command=f"""AWS_DEFAULT_REGION=$REGION \
        torchx run --workspace '' -s aws_batch \
        -cfg queue={JOB_QUEUE_GPU},image_repo={ECR_URL}/charnn dist.ddp \
        --image {ECR_URL}/charnn:latest \
        --cpu 8 --gpu 2 -j 1x2 --memMB 20480 \
        --script main.py trainer.work_dir={work_dir} trainer.job_name={job_name} \
        2>&1 | grep -Eo aws_batch://torchx/{JOB_QUEUE_GPU}:main-[a-z0-9]+""",
    env={
        "REGION": REGION,
        "JOB_QUEUE": JOB_QUEUE_GPU,
        "ECR_URL": ECR_URL,
    },
    append_env=True,
    dag=dag,
    do_xcom_push=True,
)


def wait_for_batch_job(**context) -> bool:
    session = boto3.session.Session()
    client = session.client("batch", region_name=REGION)
    # XComs are a mechanism that let Tasks talk to each other
    # Learn more from https://airflow.apache.org/docs/apache-airflow/stable/concepts/xcoms.html
    job = context["ti"].xcom_pull(task_ids="train")
    job_desc = job.split("/")[-1]
    queue_name, job_name = job_desc.split(":")
    job_id = client.list_jobs(
        jobQueue=queue_name,
        filters=[{"name": "JOB_NAME", "values": [job_name]}],
    )["jobSummaryList"][0]["jobId"]
    waiter = BatchWaitersHook(region_name=REGION)
    waiter.wait_for_job(job_id)
    return waiter.check_job_success(job_id)


wait_for_job = PythonOperator(
    task_id="wait_for_job",
    python_callable=wait_for_batch_job,
    dag=dag,
)


export = BatchOperator(
    task_id="export",
    job_name="charnn-export-job",
    job_queue=JOB_QUEUE_GENERAL,
    job_definition=JOB_DEFINITION_GENERAL,
    overrides={
        "command": f"python export.py --input_path {module_path} --output_path {exported_module_path} --torchscript".split()
    },
    dag=dag,
)

deploy = BatchOperator(
    task_id="deploy",
    job_name="charnn-deploy-job",
    job_queue=JOB_QUEUE_GENERAL,
    job_definition=JOB_DEFINITION_GENERAL,
    overrides={"command": f"./serve/deploy.sh {exported_module_path}".split()},
    dag=dag,
)

train >> wait_for_job >> export >> deploy

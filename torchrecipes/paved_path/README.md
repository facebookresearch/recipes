# paved path project

**This project is currently in Prototype. If you have suggestions for improvements, please open a GitHub issue. We'd love to hear your feedback.**

## Local development
1. Install dependencies
```bash
pip install -r requirements.txt
```

2. Model training
* Train a model
```bash
python charnn/main.py
```
You will get output like below. The snapshot path can be used for inference or restore training
```
0: epoch 0 iter 100: train loss 3.01963
0: epoch 0 iter 200: train loss 2.69831
0: epoch 0 iter 0: test loss 2.67962
0: epoch 0 iter 100: test loss 2.69960
0: epoch 0 iter 200: test loss 2.70585
...
[2022-08-30 20:07:33,842][trainer][INFO] - Saving snapshot to /tmp/charnn/run-bc6565c7/snapshots/epoch-1
```
* Restore from snapshot and train with more epochs
```bash
python charnn/main.py trainer.max_epochs=3 trainer.snapshot_path=/tmp/charnn/run-1f7abaed/snapshots/epoch-1
```

3. Generate text from a model
```bash
python charnn/main.py charnn.task="generate" charnn.phrase="hello world" trainer.snapshot_path=/tmp/charnn/run-1f7abaed/snapshots/epoch-1
```

4. [Optional] train a model with torchx
```bash
torchx run  -s local_cwd dist.ddp -j 1x2 --script charnn/main.py
```
* NOTE: 
    * `-j 1x2` means single node with 2 GPUs. Learn more about torchx [here](https://pytorch.org/torchx/latest/)

## Development in AWS
### Setup environment
1. Launch an EC2 instance following [EC2 GetStarted](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/EC2_GetStarted.html)
2. Install docker and nvidia driver if not already installed

You can use EC2 for [Local development](#Local-development). However, you may need to a cluster and scheduler to manage resources(GPU, CPU, RAM, etc.) more efficiently. There are various options like [Slurm](https://slurm.schedmd.com/documentation.html), [kubernetes](https://kubernetes.io/), etc. AWS provides a fully managed [Batch](https://aws.amazon.com/batch/) that is easy to get started. We will use it as the default scheduler in this example. With torchx, the job launching CLI will be similar for all supported schedulers.

### Create a container image on AWS ECS
Before launching a job in Batch, we need to create a docker image containing the executable(`charnn/main.py` and its dependencies). Please follow [docker/README.md](https://github.com/facebookresearch/recipes/tree/main/torchrecipes/paved_path/docker).

### AWS Batch
1. Create Batch through Wizard: https://docs.aws.amazon.com/batch/latest/userguide/Batch_GetStarted.html
  * NOTE: 
    * Configure Compute Environment and Job Queue(named it as "torch-gpu"). Do not need to Define Job if launch with torch.x
2. Setup env variables
```bash
export REGION="us-west-2"  # or any region in your case
export JOB_QUEUE="torchx-gpu"  # must match the name of your Job Queue
export ECR_URL="YOUR_AWS_ACCOUNT_ID.dkr.ecr.YOUR_REGION.amazonaws.com/charnn"  # defined in docker/README.md
```
3. Launch a model training job with torchx
```bash
torchx run --workspace '' -s aws_batch \
        -cfg queue=$JOB_QUEUE,image_repo=$ECR_URL/charnn dist.ddp \
        --script charnn/main.py --image $ECR_URL/charnn:latest \
        --cpu 8 --gpu 2 -j 1x2 --memMB 20480
```
Note that it will output a URL like "aws_batch://torchx/..." that is used to track the job status.
4. Check job status
```bash
torchx status "aws_batch://torchx/..."
```

## Pipelines
As your applications getting complicated, you can make them as pipelines, manage and monitor them by frameworks like Airflow, Kubeflow, etc.
* [Airflow example](https://github.com/facebookresearch/recipes/tree/main/torchrecipes/paved_path/airflow)
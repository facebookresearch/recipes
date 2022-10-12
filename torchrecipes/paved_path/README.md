# paved path project

**This project is currently in Prototype. If you have suggestions for improvements, please open a GitHub issue. We'd love to hear your feedback.**

## Local development
1. Install dependencies
```bash
pip install -r requirements.txt
```

2. Model training
* Train a GPT model for char generation task.
```bash
cd charnn
python main.py trainer.job_name=my_job
```
You will get output like below. The snapshot path can be used for inference or restore training
```
0: epoch 0 iter 100: train loss 3.01963
0: epoch 0 iter 200: train loss 2.69831
0: epoch 0 iter 0: test loss 2.67962
0: epoch 0 iter 100: test loss 2.69960
0: epoch 0 iter 200: test loss 2.70585
...
[2022-10-12 21:30:50,161][trainer][INFO] - Saving snapshot to /tmp/charnn/my_job/snapshots/last
[2022-10-12 21:30:50,161][__main__][INFO] - Saving CombinedModule to /tmp/charnn/my_job/modules/last.pt
```
* [Optional] train the model with torchx. It's especially useful for multi-node multi-GPU training and
you can switch between various clusters/schedulers easily.
```bash
torchx run  -s local_cwd dist.ddp -j 1x2 --script main.py
```
> **_NOTE_**: `-j 1x2` specifies single node with 2 GPUs. Learn more about torchx [here](https://pytorch.org/torchx/latest/)

* Resume training from a snapshot and train with more epochs
```bash
python main.py trainer.max_epochs=3 trainer.snapshot_path=/tmp/charnn/my_job/snapshots/last
```

3. Model transformation.
* Export the module with torchscript. Optionally, you can append `--quantize` to quantize the model.
```bash
python export.py --input_path /tmp/charnn/my_job/modules/last.pt --output_path /tmp/charnn/my_job/modules/export.pt --torchscript
```

4. Deployment
* Archive the module with its handler.py such that torchserve knows how to load and inference with the model
```bash
cd ../serve
mkdir model_store
torch-model-archiver --model-name gpt --version 1.0 --serialized-file  /tmp/charnn/my_job/modules/exported.pt --handler handler.py --export-path model_store
```
It will generate a archive file at "model_store/gpt.mar"

* Start torchserve.
```bash
torchserve --start --model-store model_store --models all
```
It will start a server at http://127.0.0.1:8080

* Test model serving with curl
```bash
curl http://127.0.0.1:8080/predictions/gpt -H "Content-Type: text/plain" --data "hello world"
```
Example response:
```
hello world sh wor m hete hes asases b bhe, he th by
```

* Stop torchserve
```bash
torchserve --stop
```


## Development in AWS
### Setup environment
1. Launch an EC2 instance following [EC2 GetStarted](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/EC2_GetStarted.html)
2. Install docker and nvidia driver if not already installed

You can use EC2 for [Local development](#Local-development). However, you may need to a cluster and scheduler to manage resources(GPU, CPU, RAM, etc.) more efficiently. There are various options like [Slurm](https://slurm.schedmd.com/documentation.html), [kubernetes](https://kubernetes.io/), etc. AWS provides a fully managed [Batch](https://aws.amazon.com/batch/) that is easy to get started. We will use it as the default scheduler in this example. With torchx, the job launching CLI will be similar for all supported schedulers.

### Create a container image on AWS ECS
Before launching a job in Batch, we need to create a docker image containing the executable(`charnn/main.py` and its dependencies). Please follow [docker/README.md](https://github.com/facebookresearch/recipes/tree/main/torchrecipes/paved_path/docker).

### AWS Batch
1. Create Batch through Wizard: https://docs.aws.amazon.com/batch/latest/userguide/Batch_GetStarted.html
  > **_NOTE_**: Configure Compute Environment and Job Queue(named it as "torch-gpu"). Do not need to Define Job if launch with torch.x
2. Setup env variables
```bash
export REGION="us-west-2"  # or any region in your case
export JOB_QUEUE="torchx-gpu"  # must match the name of your Job Queue
export ECR_URL="YOUR_AWS_ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/charnn"  # defined in docker/README.md
```
3. Launch a model training job with torchx
```bash
torchx run --workspace '' -s aws_batch \
        -cfg queue=$JOB_QUEUE,image_repo=$ECR_URL/charnn dist.ddp \
        --script charnn/main.py --image $ECR_URL/charnn:latest \
        --cpu 8 --gpu 2 -j 1x2 --memMB 20480
```
You should get output as below. You can monitor and manage the job in AWS Batch console through the `UI URL`.
```
torchx 2022-08-29 22:01:22 INFO     Found credentials in environment variables.
aws_batch://torchx/torchx-gpu:main-pqwtnnj6dqhr0
torchx 2022-08-29 22:01:23 INFO     Launched app: aws_batch://torchx/torchx-gpu:main-pqwtnnj6dqhr0
torchx 2022-08-29 22:01:23 INFO     AppStatus:
    State: PENDING
    Num Restarts: -1
    Roles:
    Msg: <NONE>
    Structured Error Msg: <NONE>
    UI URL: https://us-west-2.console.aws.amazon.com/batch/home?region=us-west-2#jobs/mnp-job/d6c98cdd-693e-47b8-981b-69e119743768
```

## Pipelines
As your applications getting complicated, you can make them as pipelines, manage and monitor them by frameworks like Airflow, Kubeflow, etc.
* [Airflow example](https://github.com/facebookresearch/recipes/tree/main/torchrecipes/paved_path/airflow)

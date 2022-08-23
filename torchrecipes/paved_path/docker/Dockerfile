FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

# Set working directory
WORKDIR /workspace/paved_path

COPY requirements.txt .

# Install Requirements
ARG aws
RUN if [ "$aws" = true ]; then pip3 install boto3==1.21.21; fi
RUN apt-get update && apt-get upgrade -y
RUN pip3 install -r requirements.txt

# Copy training script
COPY ./charnn .


FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

# Set working directory
WORKDIR /workspace/paved_path

COPY requirements.txt .

# Install Requirements
ARG aws
RUN if [ "$aws" = true ]; then pip3 install boto3==1.21.21; fi
RUN apt-get update && apt-get upgrade -y
RUN pip3 install -r requirements.txt

# Install awscli v2
RUN apt-get install curl unzip -y
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && unzip awscliv2.zip && ./aws/install

# Install jdk-11. It's required by torchserve
RUN apt-get install openjdk-11-jdk -y

# Copy training script
COPY ./charnn .

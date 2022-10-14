#!/bin/bash
S3_URL=$1
LOCAL_MODULE_PATH="/tmp/charnn/exported.pt"

# Set working directory to serve/
cd "$(dirname "$0")" || exit

# Download the exported module file from s3
aws s3 cp "$S3_URL" "$LOCAL_MODULE_PATH"

# Archive the module file with its handler and save to "model_store/gpt.mar"
mkdir -p model_store
torch-model-archiver --model-name gpt --version 1.0 --serialized-file $LOCAL_MODULE_PATH --handler handler.py --export-path model_store --force

# start torchserve with "model_store/gpt.mar"
torchserve --start --model-store model_store --models gpt

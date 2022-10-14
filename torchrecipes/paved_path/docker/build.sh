#!/bin/bash

sudo docker build -f docker/Dockerfile -t charnn:latest ./ --build-arg aws=true
sudo docker tag charnn:latest "$ECR_URL/charnn:latest"
sudo docker push "$ECR_URL/charnn:latest"

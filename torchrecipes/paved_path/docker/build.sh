#!/bin/bash

docker build -f docker/Dockerfile -t charnn:latest ./ --build-arg aws=true
docker tag charnn:latest $ECR_URL/charnn:latest
docker push $ECR_URL/charnn:latest

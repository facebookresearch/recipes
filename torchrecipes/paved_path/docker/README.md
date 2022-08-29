## Build and push a docker image

1. Authenticate to your default registry
```bash
export REGION=YOUR_AWS_REGION
export ECR_URL=YOUR_AWS_ACCOUNT_ID.dkr.ecr.region.amazonaws.com
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ECR_URL
```
2. Create a repository
```bash
aws ecr create-repository \
    --repository-name charnn \
    --image-scanning-configuration scanOnPush=true \
    --region $REGION
```
3. Build and push the image to Amazon ECR
```bash
cd recipes/torchrecipes/paved_path
./docker/build.sh
```

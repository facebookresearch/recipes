# Image Classification Training Recipe

Recipe for training image classification models from TorchVision.

## Training

### Launch Jobs without TorchX

* We can easily launch a job to train a ResNet18 model on the CIFAR10 dataset with the following command:

    ```bash
    python torchrecipes/vision/image_classification/main.py
    ```

* Config overrides allow us to swap out different parts of the training job. For example, the following command launches a job to train a ResNet50 model on GPUs:

    ```bash
    python torchrecipes/vision/image_classification/main.py --config-name default_config module/model=resnet50 trainer=gpu
    ```

### Launch Jobs with TorchX

* We often use [TorchX](https://pytorch.org/torchx) to launch training jobs across different environments. You can install TorchX with

    ```bash
    pip install torchx
    ```

* Training jobs can then be launched with the following commands:

    ```bash
    torchx run --scheduler local_cwd utils.python --script torchrecipes/vision/image_classification/main.py
    ```

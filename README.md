[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](LICENSE)

 # TorchRecipes

<h3 align='center'>Train machine learning models with a couple of lines of code with <code>torchrecipes</code>.</h3>

> This library is currently under heavy development - if you have suggestions on the API or use-cases you'd like to be covered, please open an GitHub issue or reach out. We'd love to hear about how you're using `torchrecipes`!

A recipe is a ready-to-run application that trains a deep learning model by combining a model architecture, trainer, config, etc that you can easily run, modify, or extend. Recipes run on everything from local development environments on your laptop, to large scale clusters. They enable quick experimentation through configuration, and a good starting place to extend the code by forking when more extensive changes are needed.

We provide a number of out-of-the-box recipes across popular domains (vision, NLP, etc) and tasks (image classification, etc) that you can use immediately or as a starting point for further work.

## Why `torchrecipes`?

Getting started with training machine learning models is a lot easier if you can start with something that already runs, instead of having to write all the glue code yourself.

Machine learning, whether for research or production training, requires working with a number of components like training loops or frameworks, configuration/hyper-parameter parsing, model architectures, data loading, etc. Recipes provide production-ready examples for common tasks that can be easily modified. A recipe at a high-level integrates these modular components so that you can modify the ones that matter for your problem!

We focus our recipes on providing consistent, high-quality baselines that accurately reproduce research papers.

## Supported Tasks

### Vision

- [Image Classification (using TorchVision ResNets)](torchrecipes/vision/image_classification)

## Anatomy of a Recipe

A recipe is a configurable Python application that you can run directly or customize:

* `train_app.py`: the entrypoint - an application that will start training
* `callbacks/`: callbacks executed as part of the training loop
* `conf/`: Hydra configuration for the training job (including defaults)
* `metrics/`: metrics needed for this task
* `module/`: the model implementation
* `tests/`: tests for recipe

By default each recipe supports changing common things via configuration files like hyper-parameters as well as changing the model that is loaded itself (e.g. change from `ResNet18` to `ResNet52`). For research and experimentation, you can also override any of the functionality directly by modifying the model, training loop, etc.

### Launching standalone

To run a recipe you can launch it standalone:

```
CONFIG_MODULE="torchrecipes.vision.image_classification.conf" \
  MODE="prod" \
  HYDRA_FULL_ERROR="1" \
  python -m torchrecipes.launcher.run --config-name train_app --config-path "torchrecipes/vision/image_classification/conf"
```

### Launching with `TorchX`

TorchX makes it easy to run workflows seamlessly moving between environments (i.e. whether launching locally, on a cluster, or through a cloud provider). It is the recommended way to use `TorchRecipes`.

```
pip install torchx
```

Then go to `torchrecipes/launcher/` and create a file `torchx_app.py`:

```
# 'torchrecipes/launcher/torchx_app.py'

import torchx.specs as specs

image_classification_args = [
    "-m", "run",
    "--config-name",
    "train_app",
    "--config-path",
    "torchrecipes/vision/image_classification/conf",
]

def torchx_app(image: str = "run.py:latest", *job_args: str) -> specs.AppDef:
    return specs.AppDef(
        name="run",
        roles=[
            specs.Role(
                name="run",
                image=image,
                entrypoint="python",
                args=[*image_classification_args, *job_args],
                env={
                    "CONFIG_MODULE": "torchrecipes.vision.image_classification.conf",
                    "MODE": "prod",
                    "HYDRA_FULL_ERROR": "1",
                }
            )
        ],
    )

```

This app defines the entrypoint, args and image for launching.

Now that we have created a torchx app, we are (almost) ready for launching a job!

Firstly, create a symlink for `launcher/run.py` at the top level of the repo:

```
ln -s torchrecipes/launcher/run.py ./run.py
```

Then we are ready-to-go! Simply launch the image_classification recipe with the following command:

```
torchx run --scheduler local_cwd torchrecipes/launcher/torchx_app.py:torchx_app trainer.fast_dev_run=True trainer.checkpoint_callback=False +tb_save_dir=/tmp/
```


## Requirements

PyTorch Recipes (torchrecipes):

* python3 (3.8+)
* torch

## Installation

We recommend Anaconda as Python package management system. Please refer to pytorch.org for the detail of PyTorch (torch) installation.

```bash
# install torchrecipes
pip install torchrecipes
```

To install `torchrecipes` from source, please run the following commands:

```bash
git clone https://github.com/facebookresearch/recipes.git && cd recipes
pip install -e .
```

## Contributing

We welcome PRs! See the [CONTRIBUTING](CONTRIBUTING.md) file.

## License

torchrecipes is BSD licensed, as found in the [LICENSE](LICENSE) file.

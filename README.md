[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](LICENSE)

 # torchrecipes

**This library is currently under heavy development - if you have suggestions on the API or use-cases you'd like to be covered, please open an github issue or reach out. We'd love to hear about how you're using `torchrecipes`.**

`torchrecipes` is a prototype is built on top of PyTORCH and provides a set of reproduci-able, re-usable, ready-to-run RECIPES for training different types of models, across multiple domains, on PyTorch Lightning.

It aims to provide reproduci-able "applications" built on top of PyTorch with good performance and easy reproduciability. Because this project builds on the pytorch ecosystem and requires significant investment, we'd love to hear from and work with early adopters to shape the design. Please reach out on the issue tracker if you're interested in using this for your project.


## Why `torchrecipes`?

The primary goal of the `torchrecipes` is to 10x ML development by providing standard blueprints to easily train production-ready ML models across environemnts (from local development to cluster deployment).

## Requirements
PyTorch Recipes (torchrecipes):
* python3 (3.8+)
* torch

## Running

The easiest way to run torchrecipes is to use torchx. You can install it directly (if not already included as part of our requirements.txt) with:

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

### Release

```bash
# install torchrecipes
pip install torchrecipes
```

## Contributing

We welcome PRs! See the [CONTRIBUTING](CONTRIBUTING.md) file.

## License

torchrecipes is BSD licensed, as found in the [LICENSE](LICENSE) file.

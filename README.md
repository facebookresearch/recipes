[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](LICENSE)

 # TorchRecipes

<h3 align='center'>Train machine learning models with a couple of lines of code with <code>torchrecipes</code>.</h3>

> This library is currently under heavy development - if you have suggestions on the API or use-cases you'd like to be covered, please open an GitHub issue or reach out. We'd love to hear about how you're using `torchrecipes`!

A recipe is a ready-to-run application that trains a deep learning model by combining a model architecture, trainer, config, etc that you can easily run, modify, or extend. Recipes run on everything from local development environments on your laptop, to large scale clusters. They enable quick experimentation through configuration, and a good starting place to extend the code by forking when more extensive changes are needed.

We provide a number of out-of-the-box recipes across popular domains (vision, NLP, etc) and tasks (image, text classification, etc) that you can use immediately or as a starting point for further work.

## Why `torchrecipes`?

Getting started with training machine learning models is a lot easier if you can start with something that already runs, instead of having to write all the glue code yourself.

Machine learning, whether for research or production training, requires working with a number of components like training loops or frameworks, configuration/hyper-parameter parsing, model architectures, data loading, etc. Recipes provide production-ready examples for common tasks that can be easily modified. A recipe at a high-level integrates these modular components so that you can modify the ones that matter for your problem!

We focus our recipes on providing consistent, high-quality baselines that accurately reproduce research papers.

## Get Started

### Installation

We recommend Anaconda as Python package management system. Please refer to pytorch.org for the detail of PyTorch (torch) installation.

```bash
pip install torchrecipes
```

To install `torchrecipes` from source, please run the following commands:

```bash
git clone https://github.com/facebookresearch/recipes.git && cd recipes
pip install -e .
```

### Vision

- [Image Classification Recipe](torchrecipes/vision/image_classification)

### Text
- [Text Classification Recipe](torchrecipes/text/doc_classification)


## Anatomy of a Recipe

A recipe is a Python application that you can run directly or customize:

* `main.py`: the entrypoint to start training. The script name doesn't matter and might be different in various recipes.
* `conf/`: Hydra configuration for the training job (including defaults)
* `module/`: the model implementation for pytorch-lightning based recipes
* `data_module/`: the data loading/processing implementation for pytorch-lightning based recipes
* `tests/`: tests for recipe

By default each recipe supports changing common things via configuration files like hyper-parameters as well as changing the model that is loaded itself (e.g. change from `ResNet18` to `ResNet52`). For research and experimentation, you can also override any of the functionality directly by modifying the model, training loop, etc.


## Contributing

We welcome PRs! See the [CONTRIBUTING](CONTRIBUTING.md) file.

## License

TorchRecipes is BSD licensed, as found in the [LICENSE](LICENSE) file.

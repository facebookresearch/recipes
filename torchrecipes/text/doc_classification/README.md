# Text Classification Training Recipe

Recipe for fine-tuning a pre-trained XLM-R model for text classification using the SST-2 dataset.

### Background
#### Task
Supervised text classification is the problem of categorizing a piece of text into one or more classes from a set of predefined classes. The text can be of arbitrary length: a character, a word, a sentence, a paragraph, or a full document. For this recipe, we are doing binary classification since the SST-2 dataset only has 2 labels.

#### Dataset
The [Stanford Sentiment Treebank SST-2](https://aclanthology.org/D13-1170/) dataset contains 215,154 phrases with fine-grained sentiment labels in the parse trees of 11,855 sentences from movie reviews. Model performances are evaluated either based on a fine-grained (5-way) or binary classification model based on accuracy. The SST-2 dataset used in this recipe has binary labels (0, 1) representing positive or negative sentiment for each phrase.

[Papers With Code link](https://paperswithcode.com/dataset/sst)

#### Model
The XLM-RoBERTa model was proposed in [Unsupervised Cross-lingual Representation Learning at Scale](https://arxiv.org/abs/1911.02116). It is a large multi-lingual language model, trained on 2.5TB of filtered CommonCrawl data and based on the RoBERTa model architecture. In this recipe, we use a pre-trained XLM-R model and fine-tune it for text classification on the SST-2 dataset.

[Papers With Code link](https://paperswithcode.com/paper/unsupervised-cross-lingual-representation-1)

## Training
* Train with [default config](https://github.com/facebookresearch/recipes/blob/main/torchrecipes/text/doc_classification/conf/default_config.yaml)(with XLM-R model and SST-2 dataset using CPU):
```
$ python main.py
```
* It's time-consuming to complete the full training. For quick debugging, you can override trainer config option `fast_dev_run`
```
$ python main.py trainer.fast_dev_run=true
```
* Train with single GPU
```
$ python main.py trainer=single_gpu
```
* Train with multiple GPUs(default to 8 GPUs). Note that we need to Download the SST-2 dataset locally. This step is temporary until the [torchdata multiprocessing issue](https://github.com/pytorch/data/issues/144) gets resolved.
```
$ mkdir -p ~/.torchtext/cache/SST2 # make SST2 dataset folder
$ wget -P ~/.torchtext/cache/SST2 https://dl.fbaipublicfiles.com/glue/data/SST-2.zip # download the dataset

$ python main.py trainer=multi_gpu
```
* Train with 2 GPUs
```
$ python main.py trainer=multi_gpu trainer.devices=2
```
* Switch to a different model by overriding `module/model` config group.
```
$ python main.py module/model=xlmrbase_classifier_tiny
```
* You can load a different default config file([tiny_model_full_config.yaml](https://github.com/facebookresearch/recipes/blob/main/torchrecipes/text/doc_classification/conf/tiny_model_full_config.yaml)) by specifying `--config-name`
```
$ python main.py --config-name=tiny_model_full_config
```
*This recipe uses Hydra config. You can learn more from [Hydra Getting Started](https://hydra.cc/docs/intro/)*

## Training with torchx
[TorchX](https://pytorch.org/torchx/0.2.0dev0/) is a universal job launcher for PyTorch applications. Optionally, you can use torchx to launch this recipe with various schedulers(Local, Docker, Kubernetes, Slurm, etc.).

#### Install torchx
```
$ pip install torchx
```
#### train locally with single GPU
```
$ torchx run --scheduler local_cwd utils.python --script main.py trainer=single_gpu
```

#### train locally with multiple GPUs

```
$ mkdir -p ~/.torchtext/cache/SST2 # make SST2 dataset folder
$ wget -P ~/.torchtext/cache/SST2 https://dl.fbaipublicfiles.com/glue/data/SST-2.zip # download the dataset

torchx run --scheduler local_cwd utils.python --script main.py trainer=multi_gpu trainer.devices=2
```

#### train remotely with SLURM
First request an interactive host with specified gpus using SLURM:
```
$ srun -p dev -t 3:00:00 --gres=gpu:2 --cpus-per-task=20 --pty bash
```

Now run the following script to launch training using multiple GPUs:
```
$ torchx run --scheduler local_cwd utils.python --script main.py trainer=multi_gpu trainer.gpus=2
```

### Common Mistakes
* Ensure you have activated the environment where torchrecipes is installed (i.e. `conda activate torchrecipes`)
* You may need to update the permissions for the `\tmp` folder to allow the training recipe to save files inside the folder. Alternatively you could replace this with a folder in your home dir (i.e. `trainer.logger.save_dir=/home/{user}/tmp/`)
* You can update `trainer.gpus` parameter to use the number of GPUs available in your machine but in doing so you will also need to modify the learning rate (`module.optim.lr`) and the (`datamodule.batch_size`) parameters to ensure similar training results

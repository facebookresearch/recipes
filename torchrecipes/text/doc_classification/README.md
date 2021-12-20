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

### Preparation

In this recipe, we use TorchX to launch training job using either CPU or GPU. You can install TorchX with

```
$pip install torchx
```

Then create a file `recipes/torchrecipes/launcher/torchx_app.py` with the following contents

```
# 'recipes/torchrecipes/launcher/torchx_app.py'

import torchx.specs as specs

doc_classification_args = [
    "-m", "run",
    "--config-name",
    "train_app",
    "--config-path",
    "torchrecipes/text/doc_classification/conf",
]

def torchx_app(image: str = "run.py:latest", *job_args: str) -> specs.AppDef:
    return specs.AppDef(
        name="run",
        roles=[
            specs.Role(
                name="run",
                image=image,
                entrypoint="python",
                args=[*doc_classification_args, *job_args],
                env={
                    "CONFIG_MODULE": "torchrecipes.text.doc_classification.conf",
                    "MODE": "prod",
                    "HYDRA_FULL_ERROR": "1",
                }
            )
        ],
    )
```
The `torchx_app` above defines the entrypoint, args and image for launching. Next, we create a symlink for `launcher/run.py` at the top level of the repo:

```
cd recipes
ln -s torchrecipes/launcher/run.py ./run.py
```


### Launch Training
#### CPU
Launch a training job to train the XLM-R model on the SST-2 dataset using CPU:

```
torchx run --scheduler local_cwd `# mode`\
torchrecipes/launcher/torchx_app.py:torchx_app `# torchx entry point`\
+tb_save_dir=/tmp/ `# hydra overrides`
```
#### GPU
First request an interactive host with specified gpus using SLURM:
```
srun -p dev -t 3:00:00 --gres=gpu:2 --cpus-per-task=20 --pty bash
```

Next download the SST-2 dataset locally. This step is temporary until the [torchdata multiprocessing issue](https://github.com/pytorch/data/issues/144) gets resolved.
```
mkdir -p ~/.torchtext/cache/SST2 # make SST2 dataset folder
wget -P ~/.torchtext/cache/SST2 https://dl.fbaipublicfiles.com/glue/data/SST-2.zip # download the dataset
```

Now run the following script to launch training using multiple GPUs:
```
torchx run --scheduler local_cwd  `# mode` \
torchrecipes/launcher/torchx_app.py:torchx_app `# torchx entry point` \
trainer=multi_gpu trainer.gpus=2 +tb_save_dir=/tmp/ `# hydra overrides`
```

### Common Mistakes
* Ensure you have activated the environment where torchrecipes is installed (i.e. `conda activate torchrecipes`)
* You may need to update the permissions for the `\tmp` folder to allow the training recipe to save files inside the folder. Alternatively you could replace this with a folder in your home dir (i.e. `+tb_save_dir=/home/{user}/tmp/`)
* You can update `trainer.gpus` parameter to use the number of GPUs available in your machine but in doing so you will also need to modify the learning rate (`module.optim.lr`) and the (`datamodule.batch_size`) parameters to ensure similar training results

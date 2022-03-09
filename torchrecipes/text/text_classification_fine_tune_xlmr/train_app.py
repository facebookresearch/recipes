# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

from argparse import ArgumentParser
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import torchtext
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim import AdamW

from torchrecipes.core.base_app import BaseApp
from torchrecipes.text.doc_classification.transform.doc_classification_text_transform import (
    DocClassificationTextTransform,
)
from torchrecipes.text.text_classification_fine_tune_xlmr.data_module import (
    DocClassificationDataModule,
)
from torchrecipes.text.text_classification_fine_tune_xlmr.module import DocClassificationModule


@dataclass
class Result:
    tensorboard_log_dir: Optional[str] = None
    best_model_path: Optional[str] = None
    test_metrics: Optional[List[Dict[str, float]]] = None


class TrainApp(BaseApp):
    def __init__(self, config: Any):
        super().__init__()
        self.config = config

        model = self._get_model(config.model_name)
        optim = (
            AdamW(
                model.parameters(),
                lr=config.lr,
                betas=(0.9, 0.999),
                eps=1.0e-08,
                weight_decay=0,
                amsgrad=False,
            ),
        )
        text_transform = DocClassificationTextTransform(
            vocab_path="https://download.pytorch.org/models/text/xlmr.vocab.pt",
            spm_model_path="https://download.pytorch.org/models/text/xlmr.sentencepiece.bpe.model",
        )
        self.module = DocClassificationModule(
            transform=text_transform,
            model=model,
            optim=optim,
            num_classes=2,
        )

        (
            train_dataset,
            val_dataset,
            test_dataset,
        ) = torchtext.datasets.sst2.SST2(root="~/.torchtext/cache")

        self.datamodule = DocClassificationDataModule(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            transform=text_transform,
            label_transform=None,
            columns=["text", "label"],
            label_column="label",
            batch_size=16,
            num_workers=0,
            drop_last=False,
            pin_memory=False,
        )

        # TODO @stevenliu: Auto switch to the manifold implementation when running from internal
        self.model_checkpoint = ModelCheckpoint(
            monitor="val_loss",
            save_weights_only=False,
        )
        callbacks = [self.model_checkpoint]
        self.logger = TensorBoardLogger("logs/")

        self.trainer = Trainer(
            max_epochs=self.config.max_epochs,
            logger=self.logger,
            callbacks=callbacks,
            limit_train_batches=self.config.limit_train_batches,
            limit_val_batches=self.config.limit_val_batches,
            limit_test_batches=self.config.limit_test_batches,
            log_every_n_steps=self.config.log_every_n_steps,
            fast_dev_run=self.config.fast_dev_run,
        )

    def run(self) -> Result:
        self.trainer.fit(self.module, datamodule=self.datamodule)
        if not self.config.fast_dev_run:
            test_metrics = self.trainer.test(datamodule=self.datamodule)
        else:
            test_metrics = None

        best_model_path = getattr(self.model_checkpoint, "best_model_path", None)
        return Result(
            tensorboard_log_dir=self.logger.save_dir, best_model_path=best_model_path, test_metrics=test_metrics
        )

    def _get_model(self, model_name: str):
        # sweep with various models
        if model_name == "base":
            # load pre-trained XLMR Base model
            xlmr_base = torchtext.models.XLMR_BASE_ENCODER
            classifier_head = torchtext.models.RobertaClassificationHead(
                num_classes=2, input_dim=768
            )
            model = xlmr_base.get_model(head=classifier_head)
        elif model_name == "large":
            # load pre-trained XLMR Large model
            xlmr_large = torchtext.models.XLMR_LARGE_ENCODER
            classifier_head = torchtext.models.RobertaClassificationHead(
                num_classes=2, input_dim=1024
            )
            model = xlmr_large.get_model(head=classifier_head)
        elif model_name == "custom":
            # build model from scratch with user provided checkpoint and more config options
            model = torchtext.models.RobertaModelBundle.build_model(
                encoder_conf=torchtext.models.RobertaEncoderConf(
                    vocab_size=250002,
                    embedding_dim=32,
                    num_attention_heads=1,
                    num_encoder_layers=1,
                    scaling=0.125,
                ),
                head=torchtext.models.RobertaClassificationHead(
                    num_classes=2, input_dim=32
                ),
            )
        else:
            raise ValueError(f"Invalid model type: {self.config.model}")
        return model


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--lr",
        default=1.0e-05,
        type=float,
        help="learning rate",
    )
    parser.add_argument(
        "--model_name",
        default="base",
        type=str,
        help="model to train",
    )
    parser.add_argument(
        "--max_epochs",
        default=1,
        type=int,
        help="max epoch for training",
    )
    parser.add_argument(
        "--gpus",
        default=0,
        type=str,
        help="number of gpus to use",
    )
    parser.add_argument(
        "--num_nodes",
        default=1,
        type=int,
        help="num of nodes to use",
    )
    parser.add_argument(
        "--limit_train_batches",
        default=1.0,
        type=Union[int, float],
        help="num of batches for training",
    )
    parser.add_argument(
        "--limit_val_batches",
        default=1.0,
        type=Union[int, float],
        help="num of batches for validation",
    )
    parser.add_argument(
        "--limit_test_batches",
        default=1.0,
        type=Union[int, float],
        help="num of batches for testing",
    )
    parser.add_argument(
        "--log_every_n_steps",
        default=50,
        type=int,
        help="log every n steps",
    )
    parser.add_argument(
        "--fast_dev_run",
        default=False,
        type=bool,
        help="fast run for debugging purpose",
    )

    config = parser.parse_args()

    app = TrainApp(config)
    result = app.run()
    print(result)


if __name__ == "__main__":
    main()

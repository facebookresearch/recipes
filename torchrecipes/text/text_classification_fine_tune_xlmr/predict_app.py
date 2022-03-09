# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

from argparse import ArgumentParser
from dataclasses import dataclass
from typing import Dict, List, Optional

import torchtext
from torchrecipes.core.base_app import BaseApp
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer

from torch.optim import AdamW
from torchrecipes.text.doc_classification.transform.doc_classification_text_transform import (
    DocClassificationTextTransform,
)

from torchrecipes.text.text_classification_fine_tune_xlmr.data_module import (
    DocClassificationDataModule,
)
from torchrecipes.text.text_classification_fine_tune_xlmr.module import DocClassificationModule


class PredictApp(BaseApp):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.model = torchtext.models.RobertaModelBundle.build_model(
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
            checkpoint=config.checkpoint_path,
        )
        self.text_transform = DocClassificationTextTransform(
            vocab_path="https://download.pytorch.org/models/text/xlmr.vocab.pt",
            spm_model_path="https://download.pytorch.org/models/text/xlmr.sentencepiece.bpe.model",
        )

    def run(self, input_example) -> List[str]:
        transformed_example = self.text_transform({"text": [input_example]})
        return self.model(transformed_example["token_ids"])


def main():
    parser = ArgumentParser()

    parser.add_argument(
        "--checkpoint_path",
        type=str,
        help="trained model checkpoint path",
    )
    parser.add_argument(
        "--input_example",
        type=str,
        help="input example to predict",
    )

    config = parser.parse_args()

    app = PredictApp(config)
    result = app.run(config.input_example)
    print(result)


if __name__ == "__main__":
    main()

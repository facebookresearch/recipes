from argparse import ArgumentParser
from typing import Any

import torch
from pytorch_lightning import LightningModule, LightningDataModule, Trainer
from torch.utils.data import DataLoader

from torchrecipes.core.base_app import BaseApp
from torchrecipes.demo.toy_recipes.common import RandomDataset


class ToyModule(LightningModule):
    def __init__(self, lr):
        super().__init__()
        self.lr = lr
        self.layer = torch.nn.Linear(32, 2)

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        loss = self(batch).sum()
        self.log("train_loss", loss)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        loss = self(batch).sum()
        self.log("valid_loss", loss)

    def test_step(self, batch, batch_idx):
        loss = self(batch).sum()
        self.log("test_loss", loss)

    def configure_optimizers(self):
        return torch.optim.SGD(self.layer.parameters(), lr=self.lr)


class ToyDataModule(LightningDataModule):
    def __init__(self, batch_size: int):
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(RandomDataset(32, 64), batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(RandomDataset(32, 64), batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(RandomDataset(32, 64), batch_size=self.batch_size)

    def prepare_data_per_node(self):
        pass


class TrainApp(BaseApp):
    def __init__(self, config: Any):
        super().__init__()
        self.config = config

        self.module = ToyModule(lr=self.config.lr)
        self.data_module = ToyDataModule(batch_size=self.config.batch_size)

        self.trainer = Trainer(
            max_epochs=self.config.num_epochs,
            logger=None,
        )

    def run(self):
        self.trainer.fit(model=self.module, datamodule=self.data_module)


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--num_epochs",
        default=1,
        type=int,
        help="num of epochs for training",
    )
    parser.add_argument(
        "--lr",
        default=1.0e-05,
        type=float,
        help="learning rate",
    )

    parser.add_argument(
        "--batch_size",
        default=8,
        type=int,
        help="batch size for training data",
    )
    config = parser.parse_args()

    app = TrainApp(config)
    app.run()


if __name__ == '__main__':
    main()

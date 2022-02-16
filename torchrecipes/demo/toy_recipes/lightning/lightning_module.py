import sys
import argparse
from argparse import ArgumentParser
from typing import Any, List

import torch
from pytorch_lightning import LightningModule, LightningDataModule, Trainer
from torch.utils.data import DataLoader

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

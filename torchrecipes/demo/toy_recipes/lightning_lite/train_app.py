import argparse
import sys
from argparse import ArgumentParser
from typing import Any, List

import torch
from pytorch_lightning.lite import LightningLite
from torch.utils.data import DataLoader

from torchrecipes.core.base_app import BaseApp
from torchrecipes.demo.toy_recipes.common import ToyModel, RandomDataset


class TrainApp(BaseApp, LightningLite):
    def __init__(self, config: Any):
        super().__init__()
        self.config = config

        model = ToyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=config.lr)
        self.model, self.optimizer = self.setup(model, optimizer)  # Scale your model / optimizers
        dataloader = DataLoader(RandomDataset(32, 64), batch_size=config.batch_size)
        self.dataloader = self.setup_dataloaders(dataloader)  # Scale your dataloaders

    def run(self):
        self.model.train()
        for epoch in range(self.config.num_epochs):
            for batch in self.dataloader:
                self.optimizer.zero_grad()
                logits = self.model(batch)
                loss = logits.sum()  # mock the loss for demo purpose
                self.backward(loss)  # instead of loss.backward()
                self.optimizer.step()
            print(f"epoch: {epoch}: loss: {loss}")


def get_config(argv: List[str]) -> argparse.Namespace:
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
    return parser.parse_args(argv)


def main(argv: List[str]) -> None:
    config = get_config(argv)
    app = TrainApp(config)
    app.run()


if __name__ == '__main__':
    main(sys.argv[1:])

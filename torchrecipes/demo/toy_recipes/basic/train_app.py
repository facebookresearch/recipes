import argparse
import sys
from argparse import ArgumentParser
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader

from torchrecipes.core.base_app import BaseApp
from torchrecipes.demo.toy_recipes.common import ToyModel, RandomDataset


def train(config: Any) -> Dict[str, Any]:
    model = ToyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=config.lr)
    dataloader = DataLoader(RandomDataset(32, 64), batch_size=config.batch_size)

    model.train()
    for epoch in range(config.num_epochs):
        for batch in dataloader:
            optimizer.zero_grad()
            logits = model(batch)
            loss = logits.sum()  # mock the loss for demo purpose
            loss.backward()
            optimizer.step()
        print(f"epoch: {epoch}: loss: {loss}")
    return {"loss": loss}


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
    result = train(config)
    print(f"result: {result}")


if __name__ == '__main__':
    main(sys.argv[1:])

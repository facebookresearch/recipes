import argparse
import sys
from argparse import ArgumentParser
from typing import Any, Dict, List

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader

from torchrecipes.demo.toy_recipes.common import ToyModel, RandomDataset


def train(config: Any) -> Dict[str, Any]:
    accelerator = Accelerator(cpu=True)
    model = ToyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=config.lr)
    data_loader = DataLoader(RandomDataset(32, 64), batch_size=config.batch_size)
    # Scale your model, optimizers, dataloader
    model, optimizer, data_loader = accelerator.prepare(model, optimizer, data_loader)

    # start training
    model.train()
    for epoch in range(config.num_epochs):
        for batch in data_loader:
            optimizer.zero_grad()
            logits = model(batch)
            loss = logits.sum()  # mock the loss for demo purpose
            accelerator.backward(loss)  # instead of loss.backward()
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
    """
    main function to train a model with HF Accelerate

    Usage (CLI):
        $ accelerate launch --config_file LAUNCH_CONFIG.yaml train_app.py --script_args

    Example:
        $ accelerate launch --config_file launch_config_cpu.yaml train_app.py --num_epochs 10
    """
    config = get_config(argv)
    train(config)


if __name__ == '__main__':
    main(sys.argv[1:])

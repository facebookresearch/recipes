import sys
import argparse
from argparse import ArgumentParser
from typing import Any, List

from pytorch_lightning import Trainer
from torchrecipes.demo.toy_recipes.lightning.lightning_module import ToyModule
from torchrecipes.demo.toy_recipes.lightning.data_module import ToyDataModule
from pytorch_lightning.utilities.types import _EVALUATE_OUTPUT


def train(config: Any) -> _EVALUATE_OUTPUT:
    lightning_module = ToyModule(lr=config.lr)
    data_module = ToyDataModule(batch_size=config.batch_size)

    trainer = Trainer(
        max_epochs=config.num_epochs,
        logger=None,
        gpus=config.gpus,
        num_nodes=config.num_nodes,
    )
    trainer.fit(model=lightning_module, datamodule=data_module)
    return trainer.test(model=lightning_module, datamodule=data_module)


def get_config(argv: List[str]) -> argparse.Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--mode",
        default='train',
        type=str,
        help="mode to run, available mode: train, test, predict",
    )
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
    parser.add_argument(
        "--gpus",
        default=0,
        type=int,
        help="num of GPUs",
    )
    parser.add_argument(
        "--num_nodes",
        default=1,
        type=int,
        help="num of nodes",
    )
    return parser.parse_args(argv)


def main(argv: List[str]) -> None:
    config = get_config(argv)
    result = train(config)
    print(f"result: {result}")


if __name__ == '__main__':
    main(sys.argv[1:])

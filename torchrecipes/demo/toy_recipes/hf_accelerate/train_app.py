from argparse import ArgumentParser
from typing import Any

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader

from torchrecipes.core.base_app import BaseApp
from torchrecipes.demo.toy_recipes.common import ToyModel, RandomDataset


class TrainApp(BaseApp):
    def __init__(self, config: Any):
        super().__init__()
        self.config = config

        self.accelerator = Accelerator(cpu=True)
        model = ToyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=config.lr)
        data_loader = DataLoader(RandomDataset(32, 64), batch_size=config.batch_size)
        # Scale your model, optimizers, dataloader
        self.model, self.optimizer, self.data_loader = self.accelerator.prepare(model, optimizer, data_loader)

    def run(self):
        self.model.train()
        for epoch in range(self.config.num_epochs):
            for batch in self.data_loader:
                self.optimizer.zero_grad()
                logits = self.model(batch)
                loss = logits.sum()  # mock the loss for demo purpose
                self.accelerator.backward(loss)  # instead of loss.backward()
                self.optimizer.step()
            print(f"epoch: {epoch}: loss: {loss}")


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
    args = parser.parse_args()

    app = TrainApp(args)
    app.run()


if __name__ == '__main__':
    main()

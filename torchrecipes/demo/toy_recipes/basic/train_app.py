from argparse import ArgumentParser

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from torchrecipes.core.base_app import BaseApp
from torchrecipes.demo.toy_recipe.common import ToyModel, RandomDataset


class TrainApp(BaseApp):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config

        self.model = ToyModel()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=config.lr)
        self.dataloader = DataLoader(RandomDataset(32, 64), batch_size=config.batch_size)

    def run(self):
        self.model.train()
        for epoch in range(self.config.num_epochs):
            for batch in self.dataloader:
                self.optimizer.zero_grad()
                logits = self.model(batch)
                loss = logits.sum()  # mock the loss for demo purpose
                loss.backward()
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

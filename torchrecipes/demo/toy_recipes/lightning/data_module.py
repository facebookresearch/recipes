from pytorch_lightning import LightningModule, LightningDataModule, Trainer
from torch.utils.data import DataLoader

from torchrecipes.demo.toy_recipes.common import RandomDataset


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

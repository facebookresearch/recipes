from dataclasses import dataclass
from typing import (
    Optional,
)
import pytorch_lightning as pl
from omegaconf import MISSING
from torch.utils.data import DataLoader
from torchaudio.datasets import LibriMix
from torchrecipes.core.conf import DataModuleConf
from torchrecipes.utils.config_utils import (
    config_entry,
    get_class_config_method,
)
from hydra.core.config_store import ConfigStore


class LibriMixDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root_dir: str,
        batch_size: int = 6,
        tr_split: str = "train-360",
        num_speakers: int = 2,
        sample_rate: int = 16000,
        task: str = "sep_clean",
    ) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.tr_split = tr_split
        self.num_speakers = num_speakers
        self.sample_rate = sample_rate
        self.task = task

    @config_entry
    @staticmethod
    def from_config(
        root_dir: str,
        batch_size: int = 6,
        tr_split: str = "train-360",
        num_speakers: int = 2,
        sample_rate: int = 16000,
        task: str = "sep_clean",
    ) -> "LibriMixDataModule":
        return LibriMixDataModule(
            root_dir, batch_size, tr_split, num_speakers, sample_rate, task
        )

    def setup(self, stage: Optional[str] = None):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train = LibriMix(
                self.root_dir,
                self.tr_split,
                self.num_speakers,
                self.sample_rate,
                self.task,
            )
            self.val = LibriMix(
                self.root_dir, "dev", self.num_speakers, self.sample_rate, self.task
            )

        if stage == "test" or stage is None:
            self.test = LibriMix(
                self.root_dir, "test", self.num_speakers, self.sample_rate, self.task
            )

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)


@dataclass
class LibriMixDataModuleConf(DataModuleConf):
    _target_: str = get_class_config_method(LibriMixDataModule)
    # pyre-fixme[4]: Attribute annotation cannot be `Any`.
    root_dir: str = MISSING
    batch_size: int = 6
    tr_split: str = "train-360"
    num_speakers: int = 2
    sample_rate: int = 16000
    task: str = "sep_clean"


cs = ConfigStore().instance()
cs.store(
    group="schema/datamodule",
    name="librimix_datamodule_conf",
    node=LibriMixDataModuleConf,
    package="datamodule",
)

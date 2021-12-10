from dataclasses import dataclass
from typing import (
    Optional,
)

import pytorch_lightning as pl
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from torch.utils.data import DataLoader
from torchaudio.datasets import LibriMix
from torchrecipes.core.conf import DataModuleConf
from torchrecipes.utils.config_utils import (
    config_entry,
    get_class_config_method,
)

from .test_dataset import TestDataset
from .utils import CollateFn


class LibriMixDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root_dir: str,
        batch_size: int = 6,
        tr_split: str = "train-360",
        num_speakers: int = 2,
        sample_rate: int = 8000,
        task: str = "sep_clean",
        num_workers: int = 4,
        testing: bool = False,
    ) -> None:
        """The LightningDataModule for LibriMix Dataset.
        Args:
            root_dir (str): the root directory of the dataset.
            batch_size (int, optional): the batch size of the dataset. (Default: 6)
            tr_split (str, optional): the training split in LibriMix dataset.
                Options: [``train-360`, ``train-100``] (Default: ``train-360``)
            num_speakers (int, optional): The number of speakers, which determines the directories
                to traverse. The datamodule will traverse ``s1`` to ``sN`` directories to collect
                N source audios. (Default: 2)
            sample_rate (int, optional): the sample rate of the audio. (Default: 8000)
            task (str, optional): the task of LibriMix.
                Options: [``enh_single``, ``enh_both``, ``sep_clean``, ``sep_noisy``]
                (Default: ``sep_clean``)
            num_workers (int, optional): the number of workers for each dataloader. (Default: 4)
            testing (bool, optional): To test the training recipe. If set to ``True``, the dataset will
                output random Tensors without need of the real dataset. (Default: ``False``)
        """
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.tr_split = tr_split
        self.num_speakers = num_speakers
        self.sample_rate = sample_rate
        self.task = task
        self.num_workers = num_workers
        self.testing = testing

    @config_entry
    @staticmethod
    def from_config(
        root_dir: str,
        batch_size: int = 6,
        tr_split: str = "train-360",
        num_speakers: int = 2,
        sample_rate: int = 8000,
        task: str = "sep_clean",
        num_workers: int = 8,
        testing: bool = False,
    ) -> "LibriMixDataModule":
        return LibriMixDataModule(
            root_dir,
            batch_size,
            tr_split,
            num_speakers,
            sample_rate,
            task,
            num_workers,
            testing,
        )

    def setup(self, stage: Optional[str] = None):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            if self.testing:
                self.train = TestDataset()
                self.val = TestDataset()
            else:
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
            if self.testing:
                self.test = TestDataset()
            else:
                self.test = LibriMix(
                    self.root_dir,
                    "test",
                    self.num_speakers,
                    self.sample_rate,
                    self.task,
                )

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            collate_fn=CollateFn(sample_rate=self.sample_rate, duration=3),
            num_workers=self.num_workers,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            collate_fn=CollateFn(sample_rate=self.sample_rate, duration=-1),
            num_workers=self.num_workers,
            drop_last=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            collate_fn=CollateFn(sample_rate=self.sample_rate, duration=-1),
            num_workers=self.num_workers,
        )


@dataclass
class LibriMixDataModuleConf(DataModuleConf):
    _target_: str = get_class_config_method(LibriMixDataModule)
    # pyre-fixme[4]: Attribute annotation cannot be `Any`.
    root_dir: str = MISSING
    batch_size: int = 6
    tr_split: str = "train-360"
    num_speakers: int = 2
    sample_rate: int = 8000
    task: str = "sep_clean"
    num_workers: int = 4
    testing: bool = False


cs = ConfigStore().instance()
cs.store(
    group="schema/datamodule",
    name="librimix_datamodule_conf",
    node=LibriMixDataModuleConf,
    package="datamodule",
)

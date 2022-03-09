# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


#!/usr/bin/env python3

from functools import partial
from typing import (
    Any,
    Iterable,
    List,
    Mapping,
    Optional,
    Union,
    cast,
    Callable,
)

import pytorch_lightning as pl
import torch
from pyre_extensions import none_throws
from torch.utils.data import DataLoader, IterDataPipe
from torchrec.datasets.criteo import (
    DEFAULT_CAT_NAMES,
    DEFAULT_INT_NAMES,
    DEFAULT_LABEL_NAME,
)
from torchrec.datasets.criteo import criteo_terabyte, criteo_kaggle
from torchrec.datasets.utils import rand_split_train_val, Batch
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from torchrecipes.rec.datamodules.samplers.undersampler import ProportionUnderSampler


def _transform(
    batch: Mapping[str, Union[Iterable[str], torch.Tensor]],
    num_embeddings: Optional[int] = None,
    num_embeddings_per_feature: Optional[List[int]] = None,
) -> Batch:
    cat_list: List[torch.Tensor] = []
    for col_name in DEFAULT_INT_NAMES:
        val = cast(torch.Tensor, batch[col_name])
        # minimum value in criteo 1t/kaggle dataset of int features
        # is -1/-2 so we add 3 before taking log
        cat_list.append((torch.log(val + 3)).unsqueeze(0).T)
    dense_features = torch.cat(
        cat_list,
        dim=1,
    )

    kjt_values: List[int] = []
    kjt_lengths: List[int] = []
    for (col_idx, col_name) in enumerate(DEFAULT_CAT_NAMES):
        values = cast(Iterable[str], batch[col_name])
        for value in values:
            if value:
                kjt_values.append(
                    int(value, 16)
                    % (
                        none_throws(num_embeddings_per_feature)[col_idx]
                        if num_embeddings is None
                        else num_embeddings
                    )
                )
                kjt_lengths.append(1)
            else:
                kjt_lengths.append(0)

    sparse_features = KeyedJaggedTensor.from_lengths_sync(
        DEFAULT_CAT_NAMES,
        torch.tensor(kjt_values),
        torch.tensor(kjt_lengths, dtype=torch.int32),
    )
    labels = batch[DEFAULT_LABEL_NAME]
    assert isinstance(labels, torch.Tensor)

    return Batch(
        dense_features=dense_features,
        sparse_features=sparse_features,
        labels=labels,
    )


class CriteoDataModule(pl.LightningDataModule):
    """`DataModule for Criteo 1TB Click Logs <https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset/>`_ Dataset
    Args:
        num_days: number of days (out of 25) of data to use for train/validation
            only valid for criteo 1tb, as kaggle only have 1 train file
        num_days_test: number of days (out of 25) of data to use for testing
            only valid for criteo 1tb, the test data of kaggle does not label, thus not useable
        num_embeddings: the number of embeddings (hash size) of the categorical (sparse) features
        num_embeddings_per_feature: the number of embeddings (hash size) of the categorical (sparse) features
        batch_size: int
        num_workers: int
        train_percent: percent of data to use for training vs validation- 0.0 - 1.0
        read_chunk_size: int
        dataset_name: criteo_1t or criteo_kaggle,
            note that the test dataset of kaggle does not have label
        dataset_path: Path to the criteo dataset. Users MUST pass it
        undersampling_rate: 0.0 - 1.0. Desired proportion of zero-labeled samples to
            retain (i.e. undersampling zero-labeled rows). Ex. 0.3 indicates only 30%
            of the rows with label 0 will be kept. All rows with label 1 will be kept.
            Default: None, indicating no undersampling.
        seed: Random seed for reproducibility. Default: None.
        worker_init_fn: If not ``None``, this will be called on each worker subprocess with the
            worker id (an int in ``[0, num_workers - 1]``) as input, after seeding and before data
            loading. (default: ``None``)

    Examples:
        >>> dm = CriteoDataModule(num_days=1, batch_size=3, num_days_test=1)
        >>> dm.setup()
        >>> train_batch = next(iter(dm.train_dataloader()))
    """

    def __init__(
        self,
        num_days: int = 1,
        num_days_test: int = 0,
        num_embeddings: Optional[int] = 100000,
        num_embeddings_per_feature: Optional[List[int]] = None,
        batch_size: int = 32,
        train_percent: float = 0.8,
        num_workers: int = 0,
        read_chunk_size: int = 100000,
        dataset_name: str = "criteo_1t",
        # pyre-fixme[9]: dataset_path is declared to have type `str` but is used as type `None`.
        dataset_path: str = None,
        undersampling_rate: Optional[float] = None,
        pin_memory: bool = False,
        seed: Optional[int] = None,
        worker_init_fn: Optional[Callable[[int], None]] = None,
    ) -> None:
        super().__init__()
        self._dataset_name: str = dataset_name
        self._dataset_path: str = dataset_path
        if dataset_name == "criteo_1t":
            if not (1 <= num_days <= 24):
                raise ValueError(
                    f"Dataset has only 24 days of data. User asked for {num_days} days"
                )
            if not (0 <= num_days_test <= 24):
                raise ValueError(
                    f"Dataset has only 24 days of data. User asked for {num_days_test} days"
                )
            if not (num_days + num_days_test <= 24):
                raise ValueError(
                    f"Dataset has only 24 days of data. User asked for {num_days} train days and {num_days_test} test days"
                )
        elif dataset_name != "criteo_kaggle":
            raise ValueError(
                f"Unknown dataset {self._dataset_name}. "
                + "Please choose {criteo_1t, criteo_kaggle} for dataset_name"
            )

        if not (0.0 <= train_percent <= 1.0):
            raise ValueError(f"train_percent {train_percent} must be between 0 and 1")
        if (num_embeddings is None and num_embeddings_per_feature is None) or (
            num_embeddings is not None and num_embeddings_per_feature is not None
        ):
            raise ValueError(
                "One - and only one - of num_embeddings or num_embeddings_per_feature must be set."
            )
        if num_embeddings_per_feature is not None and len(
            num_embeddings_per_feature
        ) != len(DEFAULT_CAT_NAMES):
            raise ValueError(
                f"Length of num_embeddings_per_feature ({len(num_embeddings_per_feature)}) does not match the number"
                " of sparse features ({DEFAULT_CAT_NAMES})."
            )

        # TODO handle more workers for IterableDataset

        self.batch_size = batch_size
        self._num_workers = num_workers
        self._read_chunk_size = read_chunk_size
        self._num_days = num_days
        self._num_days_test = num_days_test
        self.num_embeddings = num_embeddings
        self.num_embeddings_per_feature = num_embeddings_per_feature
        self._train_percent = train_percent
        self._undersampling_rate = undersampling_rate
        self._pin_memory = pin_memory
        self._seed = seed
        self._worker_init_fn = worker_init_fn

        self._train_datapipe: Optional[IterDataPipe] = None
        self._val_datapipe: Optional[IterDataPipe] = None
        self._test_datapipe: Optional[IterDataPipe] = None
        self.keys: List[str] = DEFAULT_CAT_NAMES

    def _create_datapipe_1t(self, day_range: Iterable[int]) -> IterDataPipe:
        # TODO (T105042401): replace the file path by using a file in memory, reference by a file handler
        paths = [f"{self._dataset_path}/day_{day}.tsv" for day in day_range]
        datapipe = criteo_terabyte(
            paths,
            # this is important because without it, the reader will attempt to synchronously download the whole file...
            read_chunk_size=self._read_chunk_size,
        )

        undersampling_rate = self._undersampling_rate
        if undersampling_rate is not None:
            datapipe = ProportionUnderSampler(
                datapipe,
                self._get_label,
                {0: undersampling_rate, 1: 1.0},
                seed=self._seed,
            )

        return datapipe

    def _create_datapipe_kaggle(self, partition: str) -> IterDataPipe:
        # note that there is no need to downsampling in in Kaggle dataset
        path = f"{self._dataset_path}/{partition}.txt"
        return criteo_kaggle(
            path,
            # this is important because without it, the reader will attempt to synchronously download the whole file...
            read_chunk_size=self._read_chunk_size,
        )

    @staticmethod
    # pyre-ignore[2, 3]
    def _get_label(row: Any) -> Any:
        return row["label"]

    def _batch_collate_transform(self, datapipe: IterDataPipe) -> IterDataPipe:
        _transform_partial = partial(
            _transform,
            num_embeddings=self.num_embeddings,
            num_embeddings_per_feature=self.num_embeddings_per_feature,
        )
        return datapipe.batch(self.batch_size).collate().map(_transform_partial)

    def setup(self, stage: Optional[str] = None) -> None:
        if self._worker_init_fn is not None:
            self._worker_init_fn(0)
        if stage == "fit" or stage is None:
            if self._dataset_name == "criteo_1t":
                datapipe = self._create_datapipe_1t(range(self._num_days))
            elif self._dataset_name == "criteo_kaggle":
                datapipe = self._create_datapipe_kaggle("train")
            else:
                raise ValueError(
                    f"Unknown dataset {self._dataset_name}. "
                    + "Please choose {criteo_1t, criteo_kaggle} for dataset_name"
                )
            train_datapipe, val_datapipe = rand_split_train_val(
                datapipe, self._train_percent
            )
            self._train_datapipe = self._batch_collate_transform(train_datapipe)
            self._val_datapipe = self._batch_collate_transform(val_datapipe)

        if stage == "test" or stage is None:
            if self._dataset_name == "criteo_1t":
                datapipe = self._create_datapipe_1t(
                    range(self._num_days, self._num_days + self._num_days_test)
                )
            elif self._dataset_name == "criteo_kaggle":
                datapipe = self._create_datapipe_kaggle("test")
            else:
                raise ValueError(
                    f"Unknown dataset {self._dataset_name}. "
                    + "Please choose {criteo_1t, criteo_kaggle} for dataset_name"
                )
            self._test_datapipe = self._batch_collate_transform(datapipe)

    def _create_dataloader(self, datapipe: IterDataPipe) -> DataLoader:
        return DataLoader(
            datapipe,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
            batch_size=None,
            batch_sampler=None,
            worker_init_fn=self._worker_init_fn,
        )

    def train_dataloader(self) -> DataLoader:
        datapipe = self._train_datapipe
        assert isinstance(datapipe, IterDataPipe)
        return self._create_dataloader(datapipe)

    def val_dataloader(self) -> DataLoader:
        datapipe = self._val_datapipe
        assert isinstance(datapipe, IterDataPipe)
        return self._create_dataloader(datapipe)

    def test_dataloader(self) -> DataLoader:
        if self._dataset_name == "criteo_1t":
            datapipe = self._test_datapipe
        elif self._dataset_name == "criteo_kaggle":
            # because kaggle test dataset does not have label
            # we use validation pipeline here
            datapipe = self._val_datapipe
        else:
            raise ValueError(
                f"Unknown dataset {self._dataset_name}. "
                + "Please choose {criteo_1t, criteo_kaggle} for dataset_name"
            )
        assert isinstance(datapipe, IterDataPipe)
        return self._create_dataloader(datapipe)

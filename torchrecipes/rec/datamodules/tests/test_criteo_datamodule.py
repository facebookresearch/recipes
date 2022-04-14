# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


#!/usr/bin/env python3


import tempfile

import testslide
from torchrecipes.rec.datamodules.criteo_datamodule import CriteoDataModule
from torchrecipes.rec.datamodules.tests.utils import (
    CAT_FEATURE_COUNT,
    create_dataset_tsv,
    INT_FEATURE_COUNT,
)


class TestCriteoDataModule(testslide.TestCase):
    def test_none_stage(self) -> None:
        num_days = 1
        num_days_test = 1
        dataset_path: str = tempfile.mkdtemp()
        with create_dataset_tsv(
            num_days=num_days, num_days_test=num_days_test, dataset_path=dataset_path
        ):
            dm = CriteoDataModule(
                num_days=num_days,
                batch_size=3,
                num_days_test=num_days_test,
                num_workers=0,
                dataset_path=dataset_path,
            )
            dm.setup()

            train_batch = next(iter(dm.train_dataloader()))
            self.assertEqual(train_batch.dense_features.size(), (3, INT_FEATURE_COUNT))
            kjt = train_batch.sparse_features
            self.assertEqual(kjt.lengths().size(), (CAT_FEATURE_COUNT * 3,))
            self.assertEqual(kjt.keys(), dm.keys)

            test_batch = next(iter(dm.test_dataloader()))
            self.assertEqual(test_batch.dense_features.size(), (3, INT_FEATURE_COUNT))
            kjt = test_batch.sparse_features
            self.assertEqual(kjt.lengths().size(), (CAT_FEATURE_COUNT * 3,))
            self.assertEqual(kjt.keys(), dm.keys)

    def test_fit_stage(self) -> None:
        num_days = 1
        num_days_test = 1
        dataset_path: str = tempfile.mkdtemp()
        with create_dataset_tsv(
            num_days=num_days, num_days_test=num_days_test, dataset_path=dataset_path
        ) as _:
            dm = CriteoDataModule(
                num_days=num_days,
                batch_size=3,
                num_days_test=num_days_test,
                num_workers=0,
                dataset_path=dataset_path,
            )
            dm.setup(stage="fit")

            train_batch = next(iter(dm.train_dataloader()))
            self.assertEqual(train_batch.dense_features.size(), (3, INT_FEATURE_COUNT))
            kjt = train_batch.sparse_features
            self.assertEqual(kjt.lengths().size(), (CAT_FEATURE_COUNT * 3,))
            self.assertEqual(kjt.keys(), dm.keys)

            with self.assertRaises(AssertionError):
                # only train/val dataloaders are set up
                dm.test_dataloader()

    def test_test_stage(self) -> None:
        num_days = 1
        num_days_test = 1
        dataset_path: str = tempfile.mkdtemp()
        with create_dataset_tsv(
            num_days=num_days, num_days_test=num_days_test, dataset_path=dataset_path
        ) as _:
            dm = CriteoDataModule(
                num_days=num_days,
                batch_size=3,
                num_days_test=num_days_test,
                num_workers=0,
                dataset_path=dataset_path,
            )
            dm.setup(stage="test")

            with self.assertRaises(AssertionError):
                # only test dataloader is set up
                dm.train_dataloader()

            test_batch = next(iter(dm.test_dataloader()))
            self.assertEqual(test_batch.dense_features.size(), (3, INT_FEATURE_COUNT))
            kjt = test_batch.sparse_features
            self.assertEqual(kjt.lengths().size(), (CAT_FEATURE_COUNT * 3,))
            self.assertEqual(kjt.keys(), dm.keys)

    def test_dataset_name(self) -> None:
        num_days = 1
        num_days_test = 1
        dataset_path: str = tempfile.mkdtemp()
        with create_dataset_tsv(
            num_days=num_days, num_days_test=num_days_test, dataset_path=dataset_path
        ) as _:
            with self.assertRaises(ValueError):
                CriteoDataModule(
                    dataset_name="bad_name",
                    num_days=num_days,
                    batch_size=3,
                    num_days_test=num_days_test,
                    num_workers=0,
                    dataset_path=dataset_path,
                )
            dm_criteo = CriteoDataModule(
                dataset_name="criteo_1t",
                num_days=num_days,
                batch_size=3,
                num_days_test=num_days_test,
                num_workers=0,
                dataset_path=dataset_path,
            )
            dm_criteo.setup()
            train_batch = next(iter(dm_criteo.train_dataloader()))
            self.assertEqual(train_batch.dense_features.size(), (3, INT_FEATURE_COUNT))
            kjt = train_batch.sparse_features
            self.assertEqual(kjt.lengths().size(), (CAT_FEATURE_COUNT * 3,))
            self.assertEqual(kjt.keys(), dm_criteo.keys)

    def test_dataset_name_kaggle(self) -> None:
        num_days = 1
        num_days_test = 1
        dataset_path: str = tempfile.mkdtemp()
        with create_dataset_tsv(
            num_days=num_days,
            num_days_test=num_days_test,
            dataset_path=dataset_path,
            is_kaggle=True,
        ) as _:
            dm_criteo = CriteoDataModule(
                dataset_name="criteo_kaggle",
                num_days=num_days,
                batch_size=3,
                num_days_test=num_days_test,
                num_workers=0,
                dataset_path=dataset_path,
            )
            dm_criteo.setup()
            train_batch = next(iter(dm_criteo.train_dataloader()))
            self.assertEqual(train_batch.dense_features.size(), (3, INT_FEATURE_COUNT))
            kjt = train_batch.sparse_features
            self.assertEqual(kjt.lengths().size(), (CAT_FEATURE_COUNT * 3,))
            self.assertEqual(kjt.keys(), dm_criteo.keys)

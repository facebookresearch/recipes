# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


#!/usr/bin/env python3

import testslide
import torch
from torchrecipes.rec.datamodules.random_rec_datamodule import RandomRecDataModule


class TestRandomRecDataModule(testslide.TestCase):
    def test_manual_seed_generator(self) -> None:
        dm1 = RandomRecDataModule(manual_seed=353434)
        iterator1 = iter(dm1.init_loader)
        dm2 = RandomRecDataModule(manual_seed=353434)
        iterator2 = iter(dm2.init_loader)

        for _ in range(10):
            batch1 = next(iterator1)
            batch2 = next(iterator2)
            self.assertTrue(torch.equal(batch1.dense_features, batch2.dense_features))
            self.assertTrue(
                torch.equal(
                    batch1.sparse_features.values(), batch2.sparse_features.values()
                )
            )
            self.assertTrue(
                torch.equal(
                    batch1.sparse_features.offsets(), batch2.sparse_features.offsets()
                )
            )
            self.assertTrue(torch.equal(batch1.labels, batch2.labels))

    def test_no_manual_seed_generator(self) -> None:
        dm1 = RandomRecDataModule()
        iterator1 = iter(dm1.init_loader)
        dm2 = RandomRecDataModule()
        iterator2 = iter(dm2.init_loader)

        for _ in range(10):
            batch1 = next(iterator1)
            batch2 = next(iterator2)
            self.assertFalse(torch.equal(batch1.dense_features, batch2.dense_features))
            self.assertFalse(
                torch.equal(
                    batch1.sparse_features.values(), batch2.sparse_features.values()
                )
            )
            # offsets not random
            self.assertTrue(
                torch.equal(
                    batch1.sparse_features.offsets(), batch2.sparse_features.offsets()
                )
            )

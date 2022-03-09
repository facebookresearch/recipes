# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


#!/usr/bin/env python3

import unittest

import pytorch_lightning as pl
from torchrec import EmbeddingBagCollection
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrecipes.rec.datamodules.random_rec_datamodule import RandomRecDataModule
from torchrecipes.rec.modules.unsharded_lightning_dlrm import UnshardedLightningDLRM


class TestUnshardedLightningDLRM(unittest.TestCase):
    def test_train_model(self) -> None:
        num_embeddings = 100
        embedding_dim = 10
        num_dense = 50

        eb1_config = EmbeddingBagConfig(
            name="t1",
            embedding_dim=embedding_dim,
            num_embeddings=num_embeddings,
            feature_names=["f1", "f3"],
        )
        eb2_config = EmbeddingBagConfig(
            name="t2",
            embedding_dim=embedding_dim,
            num_embeddings=num_embeddings,
            feature_names=["f2"],
        )
        ebc = EmbeddingBagCollection(tables=[eb1_config, eb2_config])

        model = UnshardedLightningDLRM(
            ebc,
            dense_in_features=num_dense,
            dense_arch_layer_sizes=[20, embedding_dim],
            over_arch_layer_sizes=[5, 1],
        )
        datamodule = RandomRecDataModule(num_dense=num_dense)

        trainer = pl.Trainer(
            max_epochs=3,
            enable_checkpointing=False,
            limit_train_batches=100,
            limit_val_batches=100,
            limit_test_batches=100,
            logger=False,
        )

        batch = next(iter(datamodule.init_loader))
        model(
            dense_features=batch.dense_features,
            sparse_features=batch.sparse_features,
        )
        trainer.fit(model, datamodule=datamodule)
        trainer.test(model, datamodule=datamodule)

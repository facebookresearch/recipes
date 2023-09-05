# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


#!/usr/bin/env python3

import copy
import os
import tempfile
import unittest
import uuid
from typing import List

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torch.distributed.launcher.api import elastic_launch, LaunchConfig
from torchrec import EmbeddingBagCollection
from torchrec.datasets.criteo import DEFAULT_INT_NAMES
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.test_utils import skip_if_asan
from torchrecipes.fb.utils.checkpoint import setup_checkpointing
from torchrecipes.rec.accelerators.torchrec import TorchrecStrategy
from torchrecipes.rec.datamodules.random_rec_datamodule import RandomRecDataModule
from torchrecipes.rec.modules.lightning_dlrm import LightningDLRM


def _remove_prefix(origin_string: str, prefix: str) -> str:
    if origin_string.startswith(prefix):
        return origin_string[len(prefix) :]
    else:
        return origin_string[:]


class TestLightningDLRM(unittest.TestCase):
    @classmethod
    def _run_trainer(cls) -> None:
        torch.manual_seed(int(os.environ["RANK"]))
        num_embeddings = 100
        embedding_dim = 12
        num_dense = 50
        batch_size = 3

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
        eb_configs = [eb1_config, eb2_config]

        lit_models = []
        datamodules = []
        for _ in range(2):
            ebc = EmbeddingBagCollection(tables=eb_configs, device=torch.device("meta"))
            lit_model = LightningDLRM(
                ebc,
                batch_size=batch_size,
                dense_in_features=num_dense,
                dense_arch_layer_sizes=[20, embedding_dim],
                over_arch_layer_sizes=[5, 1],
            )

            datamodule = RandomRecDataModule(
                manual_seed=564733621, num_dense=num_dense, num_generated_batches=1
            )

            lit_models.append(lit_model)
            datamodules.append(datamodule)
        lit_model1, lit_model2 = lit_models
        dm1, dm2 = datamodules

        # Load m1 state dicts into m2
        lit_model2.model.load_state_dict(lit_model1.model.state_dict())
        optim1 = lit_model1.configure_optimizers()
        optim2 = lit_model2.configure_optimizers()
        optim2.load_state_dict(optim1.state_dict())

        # train model 1 using lightning
        trainer = pl.Trainer(
            max_epochs=1,
            limit_train_batches=5,
            limit_val_batches=5,
            limit_test_batches=5,
            strategy=TorchrecStrategy(),
            accelerator=("gpu" if torch.cuda.is_available() else "cpu"),
            devices=os.environ.get("LOCAL_WORLD_SIZE", 1),
            enable_model_summary=False,
            logger=False,
            enable_checkpointing=False,
        )
        trainer.fit(lit_model1, datamodule=dm1)

        # train model 2 manually
        train_dataiterator = iter(dm2.train_dataloader())
        for _ in range(5):
            batch = next(train_dataiterator).to(lit_model2.device)
            optim2.zero_grad()
            loss, _ = lit_model2.model(batch)
            loss.backward()
            optim2.step()

        # assert parameters equal
        sd1 = lit_model1.model.state_dict()
        for name, value in lit_model2.model.state_dict().items():
            if isinstance(value, ShardedTensor):
                assert torch.equal(
                    value.local_shards()[0].tensor, sd1[name].local_shards()[0].tensor
                )
            else:
                assert torch.equal(sd1[name], value)

        # assert model evaluation equal
        test_dataiterator = iter(dm2.test_dataloader())
        with torch.no_grad():
            for _ in range(10):
                batch = next(test_dataiterator).to(lit_model2.device)
                _loss_1, (_loss_1_detached, logits_1, _labels_1) = lit_model1.model(
                    batch
                )
                _loss_2, (_loss_2_detached, logits_2, _labels_2) = lit_model2.model(
                    batch
                )
                assert torch.equal(logits_1, logits_2)

    @skip_if_asan
    def test_lit_trainer_equivalent_to_non_lit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            lc = LaunchConfig(
                min_nodes=1,
                max_nodes=1,
                nproc_per_node=2,
                run_id=str(uuid.uuid4()),
                rdzv_backend="c10d",
                rdzv_endpoint=os.path.join(tmpdir, "rdzv"),
                rdzv_configs={"store_type": "file"},
                start_method="spawn",
                monitor_interval=1,
                max_restarts=0,
            )

            elastic_launch(config=lc, entrypoint=self._run_trainer)()

    @classmethod
    def _assert_model_of_ckpt(
        cls,
        eb_configs: List[EmbeddingBagConfig],
        dense_arch_layer_sizes: str,
        over_arch_layer_sizes: str,
        checkpoint: ModelCheckpoint,
        batch_size: int,
    ) -> None:
        model1 = LightningDLRM(
            EmbeddingBagCollection(tables=eb_configs, device=torch.device("meta")),
            batch_size=batch_size,
            dense_in_features=len(DEFAULT_INT_NAMES),
            dense_arch_layer_sizes=list(map(int, dense_arch_layer_sizes.split(","))),
            over_arch_layer_sizes=list(map(int, over_arch_layer_sizes.split(","))),
        )
        model2 = LightningDLRM(
            EmbeddingBagCollection(tables=eb_configs, device=torch.device("meta")),
            batch_size=batch_size,
            dense_in_features=len(DEFAULT_INT_NAMES),
            dense_arch_layer_sizes=list(map(int, dense_arch_layer_sizes.split(","))),
            over_arch_layer_sizes=list(map(int, over_arch_layer_sizes.split(","))),
        )
        datamodule_1 = RandomRecDataModule(
            batch_size=batch_size, hash_size=64, num_dense=len(DEFAULT_INT_NAMES)
        )
        datamodule_1.setup()
        datamodule_2 = RandomRecDataModule(
            batch_size=batch_size, hash_size=64, num_dense=len(DEFAULT_INT_NAMES)
        )
        datamodule_2.setup()

        trainer = pl.Trainer(
            logger=False,
            max_epochs=3,
            callbacks=[checkpoint],
            limit_train_batches=5,
            limit_val_batches=5,
            limit_test_batches=5,
            strategy=TorchrecStrategy(),
            enable_model_summary=False,
        )

        trainer.fit(model1, datamodule=datamodule_1)

        cb_callback = trainer.checkpoint_callback
        assert cb_callback is not None
        last_checkpoint_path = cb_callback.best_model_path

        # second run

        cp_std = torch.load(last_checkpoint_path)["state_dict"]
        test_std = {}
        for name, value in cp_std.items():
            updated_name = _remove_prefix(name, "model.")
            test_std[updated_name] = copy.deepcopy(value)

        # load state dict from the chopped state_dict
        # pyre-fixme[6] Expected `collections.OrderedDict[str, torch.Tensor]` for 1st positional only
        model2.model.load_state_dict(test_std)

        # assert parameters equal
        for w0, w1 in zip(model1.model.parameters(), model2.model.parameters()):
            assert w0.eq(w1).all()
        # assert state_dict equal
        sd1 = model1.model.state_dict()
        for name, value in model2.model.state_dict().items():
            if isinstance(value, ShardedTensor):
                assert torch.equal(
                    value.local_shards()[0].tensor,
                    sd1[name].local_shards()[0].tensor,
                )
            else:
                assert torch.equal(sd1[name], value)

    @classmethod
    def _test_checkpointing(cls) -> None:
        batch_size = 32

        datamodule = RandomRecDataModule(
            batch_size=batch_size, hash_size=64, num_dense=len(DEFAULT_INT_NAMES)
        )
        keys = datamodule.keys
        embedding_dim = 8

        eb_configs = [
            EmbeddingBagConfig(
                name=f"t_{feature_name}",
                embedding_dim=embedding_dim,
                num_embeddings=64,
                feature_names=[feature_name],
            )
            for feature_idx, feature_name in enumerate(keys)
        ]
        over_arch_layer_sizes = "8,1"
        dense_arch_layer_sizes = "8,8"

        checkpoint_1 = setup_checkpointing(
            model=LightningDLRM(
                EmbeddingBagCollection(tables=eb_configs, device=torch.device("meta")),
                batch_size=batch_size,
                dense_in_features=len(DEFAULT_INT_NAMES),
                dense_arch_layer_sizes=list(
                    map(int, dense_arch_layer_sizes.split(","))
                ),
                over_arch_layer_sizes=list(map(int, over_arch_layer_sizes.split(","))),
            ),
            checkpoint_output_path=tempfile.mkdtemp(),
        )
        assert checkpoint_1 is not None

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_2 = ModelCheckpoint(dirpath=tmpdir, save_top_k=1)

            cls._assert_model_of_ckpt(
                eb_configs,
                dense_arch_layer_sizes,
                over_arch_layer_sizes,
                checkpoint_1,
                batch_size,
            )

            cls._assert_model_of_ckpt(
                eb_configs,
                dense_arch_layer_sizes,
                over_arch_layer_sizes,
                checkpoint_2,
                batch_size,
            )

    @skip_if_asan
    @unittest.skip("TODO: Fix flaky test T156681074")
    def test_checkpointing_function(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            lc = LaunchConfig(
                min_nodes=1,
                max_nodes=1,
                nproc_per_node=2,
                run_id=str(uuid.uuid4()),
                rdzv_backend="c10d",
                rdzv_endpoint=os.path.join(tmpdir, "rdzv"),
                rdzv_configs={"store_type": "file"},
                start_method="spawn",
                monitor_interval=1,
                max_restarts=0,
            )

            elastic_launch(config=lc, entrypoint=self._test_checkpointing)()

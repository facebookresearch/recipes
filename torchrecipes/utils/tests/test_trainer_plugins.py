# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


#!/usr/bin/env python3
# pyre-strict

import unittest
from typing import Union

from hydra.core.config_store import ConfigStore
from hydra.experimental import compose, initialize
from pytorch_lightning.plugins.precision import FullyShardedNativeMixedPrecisionPlugin
from pytorch_lightning.plugins.training_type import (
    Strategy,
    DDPFullyShardedStrategy,
    DDPStrategy,
)
from pytorch_lightning.trainer import Trainer
from torchrecipes.core.conf import TrainerConf
from torchrecipes.utils.trainer_plugins import get_trainer_params


def check_training_type_plugin_attribute(
    training_type_plugin: Strategy,
    attr_name: str,
    expected_val: Union[int, bool],
) -> None:
    assert hasattr(
        training_type_plugin, attr_name
    ), f"{training_type_plugin} is supposed to have attribute {attr_name}."

    assert (
        getattr(training_type_plugin, attr_name) == expected_val
    ), f"attribute {attr_name} of {training_type_plugin} is supposed to be {expected_val}."


class TestTrainerParams(unittest.TestCase):
    def test_default_trainer_conf(self) -> None:
        cs = ConfigStore.instance()
        cs.store(name="trainer", node=TrainerConf())
        with initialize():
            trainer_conf = compose(config_name="trainer")
            trainer_params = get_trainer_params(trainer_conf)
            plugins = trainer_params.get("plugins", [])
            self.assertEqual(len(plugins), 0)
            trainer = Trainer(**trainer_params)
            self.assertIsInstance(trainer, Trainer)

    def test_trainer_conf_with_find_unused_parameters_false(self) -> None:
        cs = ConfigStore.instance()
        cs.store(
            name="trainer",
            node=TrainerConf(
                num_nodes=3,
                sync_batchnorm=True,
                plugins=["ddp_find_unused_parameters_false"],
            ),
        )
        with initialize():
            trainer_conf = compose(config_name="trainer")
            trainer_params = get_trainer_params(trainer_conf)
            plugins = trainer_params.get("plugins", [])
            self.assertEqual(len(plugins), 1)
            plugin = plugins[0]
            self.assertIsInstance(plugin, DDPStrategy)
            self.assertIsNone(plugin._ddp_comm_hook)
            self.assertIsNone(plugin._ddp_comm_state)
            self.assertIsNone(plugin._ddp_comm_wrapper)
            self.assertEqual(plugin._ddp_kwargs, {"find_unused_parameters": False})
            trainer = Trainer(**trainer_params)
            self.assertIsInstance(trainer, Trainer)
            check_training_type_plugin_attribute(
                trainer.strategy,
                "num_nodes",
                3,
            )
            check_training_type_plugin_attribute(
                trainer.strategy,
                "sync_batchnorm",
                True,
            )

    def test_trainer_conf_with_ddp_fp16_compress_plugin(self) -> None:
        cs = ConfigStore.instance()
        cs.store(name="trainer", node=TrainerConf(plugins=["ddp_fp16_compress"]))
        with initialize():
            trainer_conf = compose(config_name="trainer")
            trainer_params = get_trainer_params(trainer_conf)
            plugins = trainer_params.get("plugins", [])
            self.assertEqual(len(plugins), 1)
            plugin = plugins[0]
            self.assertIsInstance(plugin, DDPStrategy)
            self.assertIsNotNone(plugin._ddp_comm_hook)
            self.assertEqual(
                plugin._ddp_comm_hook.__qualname__,
                "fp16_compress_hook",
            )
            self.assertIsNone(plugin._ddp_comm_state)
            self.assertIsNone(plugin._ddp_comm_wrapper)
            self.assertEqual(plugin._ddp_kwargs, {"find_unused_parameters": True})
            trainer = Trainer(**trainer_params)
            self.assertIsInstance(trainer, Trainer)
            check_training_type_plugin_attribute(
                trainer.strategy,
                "num_nodes",
                1,
            )
            check_training_type_plugin_attribute(
                trainer.strategy,
                "sync_batchnorm",
                False,
            )

    def test_trainer_conf_with_ddp_power_sgd_plugin(self) -> None:
        cs = ConfigStore.instance()
        cs.store(name="trainer", node=TrainerConf(plugins=["ddp_power_sgd_5k"]))
        with initialize():
            trainer_conf = compose(config_name="trainer")
            trainer_params = get_trainer_params(trainer_conf)
            plugins = trainer_params.get("plugins", [])
            self.assertEqual(len(plugins), 1)
            plugin = plugins[0]
            self.assertIsInstance(plugin, DDPStrategy)
            self.assertIsNotNone(plugin._ddp_comm_hook)
            self.assertEqual(
                plugin._ddp_comm_hook.__qualname__,
                "powerSGD_hook",
            )
            self.assertIsNotNone(plugin._ddp_comm_state)
            self.assertIsNone(plugin._ddp_comm_wrapper)
            self.assertEqual(plugin._ddp_kwargs, {"find_unused_parameters": True})
            trainer = Trainer(**trainer_params)
            self.assertIsInstance(trainer, Trainer)
            check_training_type_plugin_attribute(
                trainer.strategy,
                "num_nodes",
                1,
            )
            check_training_type_plugin_attribute(
                trainer.strategy,
                "sync_batchnorm",
                False,
            )

    def test_trainer_conf_with_ddp_fp16_compress_wrapper_power_sgd_plugin(self) -> None:
        cs = ConfigStore.instance()
        cs.store(
            name="trainer",
            node=TrainerConf(plugins=["ddp_fp16_compress_wrapper_power_sgd_5k"]),
        )
        with initialize():
            trainer_conf = compose(config_name="trainer")
            trainer_params = get_trainer_params(trainer_conf)
            plugins = trainer_params.get("plugins", [])
            self.assertEqual(len(plugins), 1)
            plugin = plugins[0]
            self.assertIsInstance(plugin, DDPStrategy)
            self.assertIsNotNone(plugin._ddp_comm_hook)
            self.assertEqual(
                plugin._ddp_comm_hook.__qualname__,
                "powerSGD_hook",
            )
            self.assertIsNotNone(plugin._ddp_comm_state)
            self.assertIsNotNone(plugin._ddp_comm_wrapper)
            self.assertEqual(
                plugin._ddp_comm_wrapper.__qualname__, "fp16_compress_wrapper"
            )
            self.assertEqual(plugin._ddp_kwargs, {"find_unused_parameters": True})
            trainer = Trainer(**trainer_params)
            check_training_type_plugin_attribute(
                trainer.strategy,
                "num_nodes",
                1,
            )
            check_training_type_plugin_attribute(
                trainer.strategy,
                "sync_batchnorm",
                False,
            )

    def test_trainer_conf_with_ddp_multiple(self) -> None:
        cs = ConfigStore.instance()
        cs.store(
            name="trainer",
            node=TrainerConf(
                plugins=["ddp_fp16_compress", "ddp_find_unused_parameters_false"]
            ),
        )
        with initialize():
            trainer_conf = compose(config_name="trainer")
            trainer_params = get_trainer_params(trainer_conf)
            plugins = trainer_params.get("plugins", [])
            self.assertEqual(len(plugins), 1)
            plugin = plugins[0]
            self.assertIsInstance(plugin, DDPStrategy)
            self.assertIsNotNone(plugin._ddp_comm_hook)
            self.assertEqual(
                plugin._ddp_comm_hook.__qualname__,
                "fp16_compress_hook",
            )
            self.assertIsNone(plugin._ddp_comm_state)
            self.assertIsNone(plugin._ddp_comm_wrapper)
            self.assertEqual(plugin._ddp_kwargs, {"find_unused_parameters": False})
            trainer = Trainer(**trainer_params)
            self.assertIsInstance(trainer, Trainer)
            check_training_type_plugin_attribute(
                trainer.strategy,
                "num_nodes",
                1,
            )
            check_training_type_plugin_attribute(
                trainer.strategy,
                "sync_batchnorm",
                False,
            )

    def test_trainer_conf_with_ddp_fully_sharded_precision_32(self) -> None:
        cs = ConfigStore.instance()
        cs.store(
            name="trainer",
            node=TrainerConf(plugins=["ddp_fully_sharded"]),
        )
        with initialize():
            trainer_conf = compose(config_name="trainer")
            trainer_params = get_trainer_params(trainer_conf)
            plugins = trainer_params.get("plugins", [])
            self.assertEqual(len(plugins), 1)
            plugin = plugins[0]
            self.assertIsInstance(plugin, DDPFullyShardedStrategy)
            trainer = Trainer(**trainer_params)
            self.assertIsInstance(trainer, Trainer)
            check_training_type_plugin_attribute(
                trainer.strategy,
                "num_nodes",
                1,
            )
            check_training_type_plugin_attribute(
                trainer.strategy,
                "sync_batchnorm",
                False,
            )

    def test_trainer_conf_with_ddp_fully_sharded_precision_16(self) -> None:
        cs = ConfigStore.instance()
        cs.store(
            name="trainer",
            node=TrainerConf(precision=16, plugins=["ddp_fully_sharded"]),
        )
        with initialize():
            trainer_conf = compose(config_name="trainer")
            trainer_params = get_trainer_params(trainer_conf)
            plugins = trainer_params.get("plugins", [])
            self.assertEqual(len(plugins), 2)
            training_type_plugin = plugins[0]
            self.assertIsInstance(training_type_plugin, DDPFullyShardedStrategy)
            precision_plugin = plugins[1]
            self.assertIsInstance(
                precision_plugin, FullyShardedNativeMixedPrecisionPlugin
            )
            trainer = Trainer(**trainer_params)
            self.assertIsInstance(trainer, Trainer)
            check_training_type_plugin_attribute(
                trainer.strategy,
                "num_nodes",
                1,
            )
            check_training_type_plugin_attribute(
                trainer.strategy,
                "sync_batchnorm",
                False,
            )

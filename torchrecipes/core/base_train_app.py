# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


#!/usr/bin/env python3

# pyre-strict

import os
import time
import traceback
from dataclasses import dataclass
from typing import Any, Dict, Optional, TypedDict, List, Tuple

import hydra
import torch
from omegaconf import OmegaConf
from pytorch_lightning import LightningDataModule, LightningModule
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks import ModelCheckpoint as OSSModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger as OSSTensorboardLogger
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.utilities.types import _EVALUATE_OUTPUT, _PREDICT_OUTPUT
from torchrecipes._internal_patches import log_run, ModelCheckpoint, TensorBoardLogger
from torchrecipes.core.conf import ModuleConf, DataModuleConf, TrainerConf
from torchrecipes.core.logger import JobStatus
from torchrecipes.utils.checkpoint import find_last_checkpoint_path
from torchrecipes.utils.trainer_plugins import get_trainer_params

OmegaConf.register_new_resolver("get_method", hydra.utils.get_method)


@dataclass
class TrainOutput:
    tensorboard_log_dir: Optional[str] = None


class TestOutput(TypedDict):
    pass


class BaseTrainApp:
    """
    A training recipe that contains all necessary parts to train a model.
    One can easily start a trainig flow with this training application.
    To use the interface, create your own TrainApp and subclass from the BaseTrainApp.
    You also need to subclass YourTaskConfig from ModuleConf.
    """

    module_conf: ModuleConf
    module: LightningModule
    datamodule_conf: Optional[DataModuleConf]
    datamodule: Optional[LightningDataModule]
    trainer_conf: TrainerConf
    log_dir: Optional[str]
    root_dir: Optional[str]

    def __init__(
        self,
        module: ModuleConf,
        trainer: TrainerConf,
        datamodule: Optional[DataModuleConf] = None,
    ) -> None:
        super().__init__()
        self.datamodule_conf = datamodule
        self.datamodule = self.get_data_module()

        self.module_conf = module
        self.module = self.get_lightning_module()

        self.trainer_conf = trainer
        self.log_dir = None
        self.root_dir = None
        torch._C._log_api_usage_once(
            f"torchrecipes.{self.__module__}.{self.__class__.__name__}"
        )

    def get_lightning_module(self) -> LightningModule:
        """
        Override this method to instantiate a LightningModule
        """
        return hydra.utils.instantiate(self.module_conf, _recursive_=False)

    def get_data_module(self) -> Optional[LightningDataModule]:
        """
        Override this method to instantiate a LightningDataModule
        """
        if self.datamodule_conf:
            return hydra.utils.instantiate(self.datamodule_conf, _recursive_=False)

        return None

    def get_callbacks(self) -> List[Callback]:
        """
        Override this method to return a list of callbacks to be passed into Trainer
        You can add additional ModelCheckpoint here
        """
        return []

    def get_logger(self) -> OSSTensorboardLogger:
        """
        Override this method to return a logger for trainer
        TODO: T88650989 set different default logger for OSS and FB TrainApp
        """
        return TensorBoardLogger()

    def get_default_model_checkpoint(self) -> OSSModelCheckpoint:
        """
        Override this method to return a default ModelCheckpoint callback.
        Note: If you want to use more than 1 ModelCheckpoint callback, add it through
        get_callbacks() function.
        """
        dirpath: Optional[str] = None
        root_dir = self.root_dir
        if root_dir:
            dirpath = os.path.join(root_dir, "checkpoints")
        return ModelCheckpoint(
            # will auto generate dirpath if not provided
            dirpath=dirpath,
            save_top_k=-1,
            has_user_data=False,
            ttl_days=1,
            monitor=None,
        )

    def _get_trainer(self) -> Tuple[Trainer, Dict[str, Any]]:
        trainer_params = self._init_trainer_params()
        self._set_trainer_params(trainer_params)

        # log trainer params
        log_params = dict(trainer_params)
        log_params["oncall_team"] = "pt_lightning"
        log_params["run_status"] = JobStatus.RUNNING.value
        log_run(**log_params)

        return Trainer(**trainer_params), log_params

    def _init_trainer_params(self) -> Dict[str, Any]:
        return get_trainer_params(self.trainer_conf)

    def _set_trainer_params(
        self,
        trainer_params: Dict[str, Any],
    ) -> None:
        # set default logger if not specified
        # if logger=False, do not add a logger
        if trainer_params.get("logger", True):
            logger = self.get_logger()
            trainer_params["logger"] = logger
            self.log_dir = logger.log_dir
            self.root_dir = logger.root_dir

        callbacks = trainer_params.get("callbacks", [])
        callbacks.extend(self.get_callbacks())

        # create default model checkpoint callback unless disabled
        if trainer_params.get("checkpoint_callback", True):
            checkpoint_callback = self.get_default_model_checkpoint()
            callbacks.append(checkpoint_callback)

            # auto-resume from last default checkpoint
            ckpt_path = checkpoint_callback.dirpath
            if not trainer_params.get("resume_from_checkpoint") and ckpt_path:
                last_checkpoint = find_last_checkpoint_path(ckpt_path)
                trainer_params["resume_from_checkpoint"] = last_checkpoint

        trainer_params["callbacks"] = callbacks

    def train(self) -> TrainOutput:
        trainer, log_params = self._get_trainer()

        start_time = time.monotonic()
        got_exception = None
        try:
            trainer.fit(self.module, datamodule=self.datamodule)
        except Exception as ex:
            got_exception = ex

        # log trainer status to Scuba and Hive
        total_run_time = int(time.monotonic() - start_time)
        log_params["global_rank"] = trainer.global_rank
        log_params["world_size"] = trainer.world_size
        log_params["total_run_time"] = total_run_time
        if got_exception is None:
            log_params["run_status"] = JobStatus.COMPLETED.value
            log_run(**log_params)
        else:
            log_params["error_message"] = str(got_exception)
            log_params["stacktrace"] = traceback.format_stack()
            log_params["run_status"] = JobStatus.FAILED.value
            log_run(**log_params)
            raise got_exception

        return TrainOutput(tensorboard_log_dir=self.log_dir)

    def test(self) -> _EVALUATE_OUTPUT:
        trainer, _ = self._get_trainer()
        return trainer.test(self.module, datamodule=self.datamodule)

    def predict(self) -> Optional[_PREDICT_OUTPUT]:
        trainer, _ = self._get_trainer()
        return trainer.predict(self.module, datamodule=self.datamodule)

#!/usr/bin/env python3
from dataclasses import dataclass
from typing import Any, List, Optional

import hydra

# TODO: Remove stl lightning import
from hydra.core.config_store import ConfigStore

# @manual "//github/third-party/omry/omegaconf:omegaconf"
from omegaconf import MISSING
from pyre_extensions import none_throws
from pytorch_lightning import LightningModule, LightningDataModule
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import TensorBoardLogger
from torchrecipes.audio.source_separation.datamodule import (
    LibriMixDataModuleConf,
)
from torchrecipes.audio.source_separation.module.conv_tasnet import (
    ConvTasNetModule,
    ConvTasNetModuleConf,
)
from torchrecipes.core.base_train_app import BaseTrainApp
from torchrecipes.core.conf import TrainerConf
from torchrecipes.core.conf.base_config import BaseTrainAppConf
from torchrecipes.utils.config_utils import get_class_name_str


class SourceSeparationTrainApp(BaseTrainApp):
    """
    This app is used to launch the image classification training / testing.
    """

    module_conf: ConvTasNetModuleConf
    datamodule_conf: LibriMixDataModuleConf

    def __init__(
        self,
        module: ConvTasNetModuleConf = ConvTasNetModuleConf(),
        trainer: TrainerConf = None,
        datamodule: LibriMixDataModuleConf = None,
        load_checkpoint_strict: bool = True,
        pretrained_checkpoint_path: Optional[str] = None,
        # pyre-fixme[2]: Parameter annotation cannot contain `Any`.
        callbacks: Optional[List[Any]] = None,
        tb_save_dir: Optional[str] = None,
    ) -> None:
        self.load_checkpoint_strict = load_checkpoint_strict
        self.pretrained_checkpoint_path = pretrained_checkpoint_path
        self.callbacks = callbacks
        self.tb_save_dir = tb_save_dir

        # This has to happen at last because it depends on the value above.
        super().__init__(module, trainer, datamodule)

    def get_data_module(self) -> Optional[LightningDataModule]:
        """
        Instantiate a LightningDataModule.
        """
        datamodule = hydra.utils.instantiate(
            self.datamodule_conf,
            _recursive_=False
        )
        return datamodule

    def get_lightning_module(self) -> LightningModule:
        """
        Instantiate a LightningModule.
        """
        # module = hydra.utils.instantiate(
        #     self.module_conf,
        #     _recursive_=False,
        # )
        module = ConvTasNetModule(
            self.module_conf.loss,
            self.module_conf.optim,
            self.module_conf.metrics,
            self.module_conf.lr_scheduler,
        )
        if self.pretrained_checkpoint_path:
            return ConvTasNetModule.load_from_checkpoint(
                checkpoint_path=none_throws(self.pretrained_checkpoint_path),
                strict=self.load_checkpoint_strict,
                model=module.model,
                loss=module.loss,
                optim=module.optim,
                metrics=module.metrics,
                lr_scheduler=module.lr_scheduler,
            )
        return module

    def get_callbacks(self) -> List[Callback]:
        """
        Override this method to return a list of callbacks to be passed
        into Trainer. You can add additional ModelCheckpoint here
        """
        if self.callbacks is None:
            return []
        else:
            return [
                hydra.utils.instantiate(callback, _recursive_=False)
                for callback in self.callbacks
            ]

    def get_logger(self) -> TensorBoardLogger:
        assert (
            self.tb_save_dir is not None
        ), "Should specify tb_save_dir if trainer.logger=True!"
        return TensorBoardLogger(save_dir=self.tb_save_dir)


@dataclass
class SourceSeparationTrainAppConf(BaseTrainAppConf):
    _target_: str = get_class_name_str(SourceSeparationTrainApp)
    module: ConvTasNetModuleConf = MISSING
    trainer: TrainerConf = MISSING
    datamodule: LibriMixDataModuleConf = MISSING
    load_checkpoint_strict: bool = True
    pretrained_checkpoint_path: Optional[str] = None
    # pyre-fixme[2]: Parameter annotation cannot contain `Any`.
    callbacks: Optional[List[Any]] = None
    tb_save_dir: Optional[str] = None


# cs: ConfigStore = ConfigStore.instance()
# cs.store(name="source_separation_app", node=SourceSeparationTrainAppConf)

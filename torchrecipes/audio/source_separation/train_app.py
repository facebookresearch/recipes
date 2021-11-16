#!/usr/bin/env python3
from dataclasses import dataclass
from typing import Any, List, Optional

import hydra

# @manual "//github/third-party/omry/omegaconf:omegaconf"
from omegaconf import MISSING
from pyre_extensions import none_throws
from pytorch_lightning import LightningModule, LightningDataModule
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import TensorBoardLogger

# TODO: Remove stl lightning import
from torchrecipes.core.base_train_app import BaseTrainApp
from torchrecipes.core.conf import TrainerConf
from torchrecipes.core.conf.base_config import BaseTrainAppConf
from torchrecipes.utils.config_utils import get_class_name_str
from torchrecipes.audio.datamodule import librimixDataModule
from torchrecipes.audio.source_separation.module.conv_tasnet import (
    ConvTasNetModule,
)


class SourceSeparationTrainApp(BaseTrainApp):
    """
    This app is used to launch the image classification training / testing.
    """

    module_conf: SourceSeparationModuleConf
    datamodule_conf: VisionDataModuleConf

    def __init__(
        self,
        module: ImageClassificationModuleConf,
        trainer: TrainerConf,
        datamodule: VisionDataModuleConf,
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
        return librimixDataModule(
            root_dir,
            batch_size,
            tr_split,
            num_speakers,
            sample_rate,
            task,
        )

    def get_lightning_module(self) -> LightningModule:
        """
        Instantiate a LightningModule.
        """
        module = hydra.utils.instantiate(
            self.module_conf,
            _recursive_=False,
        )
        if self.pretrained_checkpoint_path:
            return ConvTasNetModule.load_from_checkpoint(
                checkpoint_path=none_throws(self.pretrained_checkpoint_path),
                strict=self.load_checkpoint_strict,
                model=module.model,
                loss=module.loss,
                optim_fn=module.optim_fn,
                metrics=module.metrics,
                lr_scheduler_fn=module.lr_scheduler_fn,
                apply_softmax=module.apply_softmax,
                process_weighted_labels=module.process_weighted_labels,
                norm_weight_decay=module.norm_weight_decay,
            )
        return module

    def get_callbacks(self) -> List[Callback]:
        """
        Override this method to return a list of callbacks to be passed into Trainer
        You can add additional ModelCheckpoint here
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
    _target_: str = get_class_name_str(SourceSeparationTrainAppConf)
    datamodule: VisionDataModuleConf = MISSING
    module: ImageClassificationModuleConf = MISSING
    trainer: TrainerConf = MISSING
    pretrained_checkpoint_path: Optional[str] = None
    load_checkpoint_strict: bool = True
    # pyre-fixme[4]: Attribute annotation cannot contain `Any`.
    callbacks: Optional[List[Any]] = None
    tb_save_dir: Optional[str] = None
    

cs: ConfigStore = ConfigStore.instance()

cs.store(name="doc_classification_app", node=DocClassificationTrainAppConf)
from pathlib import Path
from argparse import ArgumentParser
from typing import (
    Any,
    Callable,
    Dict,
    Mapping,
    List,
    Optional,
    Tuple,
    Union,
)
from omegaconf import MISSING
from dataclasses import dataclass
from torchrecipes.core.conf import ModuleConf
from torch.optim.lr_scheduler import _LRScheduler
import hydra
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchaudio.models import ConvTasNet
from torchrecipes.utils.config_utils import get_class_config_method, config_entry
from torchrecipes.audio.source_separation.loss import si_sdr_loss
from torchrecipes.audio.source_separation.metrics import sdri_metric, sisdri_metric

def _get_model(
    num_sources=2,
    enc_kernel_size=16,
    enc_num_feats=512,
    msk_kernel_size=3,
    msk_num_feats=128,
    msk_num_hidden_feats=512,
    msk_num_layers=8,
    msk_num_stacks=3,
    msk_activate="relu",
):
    model = ConvTasNet(
        num_sources=num_sources,
        enc_kernel_size=enc_kernel_size,
        enc_num_feats=enc_num_feats,
        msk_kernel_size=msk_kernel_size,
        msk_num_feats=msk_num_feats,
        msk_num_hidden_feats=msk_num_hidden_feats,
        msk_num_layers=msk_num_layers,
        msk_num_stacks=msk_num_stacks,
        msk_activate=msk_activate,
    )
    return model

class ConvTasNetModule(pl.LightningModule):
    """
    The Lightning Module for speech separation.

    Args:
        loss (Any): The loss function to use.
        optim (Any): The optimizer function to use.
        metrics (List of methods): The metrics to track, which will be used for both train and validation.
        lr_scheduler (Any or None, optional): The LR Scheduler.
    """

    def __init__(
        self,
        loss: Any,
        optim: Any,
        metrics: List[Any],
        lr_scheduler: Optional[Any] = None,
    ) -> None:
        super().__init__()

        self.model: nn.Module = _get_model()
        self.loss: nn.Module = loss
        self.optim: torch.optim.Optimizer = optim
        self.lr_scheduler: Optional[_LRScheduler] = None
        if lr_scheduler:
            self.lr_scheduler = lr_scheduler

        self.metrics: Mapping[str, Callable] = metrics

        self.train_metrics: Dict = {}
        self.val_metrics: Dict = {}
        self.test_metrics: Dict = {}

        self.save_hyperparameters()

    @config_entry
    @staticmethod
    def from_config(
        loss: Any,
        optim: Any,
        metrics: List[Any],
        lr_scheduler: Optional[Any] = None,
    ) -> "ConvTasNetModule":
        model = _get_model()
        optim = hydra.utils.instantiate(
            optim,
            model.parameters(),
        )
        return ConvTasNetModule(model, loss, optim, metrics, lr_scheduler)

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit":
            self.train_metrics.update(self.metrics)
            self.val_metrics.update(self.metrics)
        else:
            self.test_metrics.update(self.metrics)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward defines the prediction/inference actions.
        """
        return self.model(x)

    def training_step(
        self,
        batch: Union[List[torch.Tensor], Mapping[str, torch.Tensor]],
        batch_idx: int,
        *args: Any,
        **kwargs: Any
    ) -> Dict[str, Any]:
        return self._step(batch, batch_idx, "train")

    def validation_step(
        self,
        batch: Union[List[torch.Tensor], Mapping[str, torch.Tensor]],
        batch_idx: int,
        *args: Any,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Operates on a single batch of data from the validation set.
        """
        return self._step(batch, batch_idx, "val")

    def test_step(
        self,
        batch: Union[List[torch.Tensor], Mapping[str, torch.Tensor]],
        batch_idx: int,
        *args: Any,
        **kwargs: Any
    ) -> Optional[Dict[str, Any]]:
        """
        Operates on a single batch of data from the test set.
        """
        return self._step(batch, batch_idx, "test")

    def _step(
        self,
        batch: Union[List[torch.Tensor], Mapping[str, torch.Tensor]],
        batch_idx: int,
        phase_type: str
    ) -> Dict[str, Any]:
        """
        Common step for training, validation, and testing.
        """
        mix, src, mask = batch
        pred = self.model(mix)
        loss = self.loss(pred, src, mask)
        self.log(f"Losses/{phase_type}_loss", loss.item(), on_step=True, on_epoch=True)

        metrics_result = self._compute_metrics(pred, src, mix, mask, phase_type)
        self.log_dict(metrics_result, on_epoch=True)

        return loss

    def configure_optimizers(
        self,
    ) -> Tuple[Any]:
        self.optim = self.optim(
            self.model.parameters,
            lr=0.001,
        )
        self.lr_scheduler = self.lr_scheduler(
            self.optim,
            mode="min",
            factor=0.5,
            patience=5
        )
        epoch_schedulers = {
            'scheduler': self.lr_scheduler,
            'monitor': 'Losses/val_loss',
            'interval': 'epoch'
        }
        return [self.optim], [epoch_schedulers]

    def _compute_metrics(
        self,
        pred: torch.Tensor,
        label: torch.Tensor,
        inputs: torch.Tensor,
        mask: torch.Tensor,
        phase_type: str,
    ) -> Dict[str, torch.Tensor]:
        metrics_dict = getattr(self, f"{phase_type}_metrics")
        metrics_result = {}
        for name, metric in metrics_dict.items():
            metrics_result[f"Metrics/{phase_type}/{name}"] = metric(pred, label, inputs, mask)
        return metrics_result

    def train_dataloader(self):
        """Training dataloader"""
        return self.train_loader

    def val_dataloader(self):
        """Validation dataloader"""
        return self.val_loader


@dataclass
class ConvTasNetModuleConf(ModuleConf):
    _target_: str = get_class_config_method(ConvTasNetModule)
    loss: Any =  si_sdr_loss
    optim: Any = torch.optim.Adam
    metrics: Any = {
        "sdri": sdri_metric,
        "sisdri": sisdri_metric,
    }
    lr_scheduler: Any = torch.optim.lr_scheduler.ReduceLROnPlateau
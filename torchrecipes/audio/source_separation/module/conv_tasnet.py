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
    TypedDict,
    Union,
)
from torch.optim.lr_scheduler import _LRScheduler
import pytorch_lightning as pl
import torch
import torch.nn as nn


class ConvTasNetModule(pl.LightningModule):
    """
    The Lightning Module for speech separation.

    Args:
        model (nn.Module): The model instance.
        train_loader (DataLoader): the training dataloader.
        val_loader (DataLoader or None): the validation dataloader.
        loss (Any): The loss function to use.
        optim (Any): The optimizer to use.
        metrics (List of methods): The metrics to track, which will be used for both train and validation.
        lr_scheduler (Any or None): The LR Scheduler.
    """

    def __init__(
        self,
        model: nn.Module,
        loss: Any,
        optim: Any,
        metrics: List[Any],
        lr_scheduler: Optional[Any] = None,
    ) -> None:
        super().__init__()

        self.model: nn.Module = model
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
        lr_scheduler = self.lr_scheduler
        if not lr_scheduler:
            return self.optim
        epoch_schedulers = {
            'scheduler': lr_scheduler,
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
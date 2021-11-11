# Copyright (c) Facebook, Inc. and its affiliates.

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from pyre_extensions import none_throws
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.trainer.trainer import Trainer
from torch.distributions.beta import Beta
from torchrecipes.utils.config_utils import get_class_name_str


def convert_to_one_hot(targets: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    This function converts target class indices to one-hot vectors,
    given the number of classes.

    """
    assert (
        torch.max(targets).item() < num_classes
    ), "Class Index must be less than number of classes"
    one_hot_targets = torch.zeros(
        (targets.shape[0], num_classes), dtype=torch.long, device=targets.device
    )
    one_hot_targets.scatter_(1, targets.long(), 1)
    return one_hot_targets


class MixupTransform(Callback):
    """
    This implements the mixup data augmentation in the paper
    "mixup: Beyond Empirical Risk Minimization" (https://arxiv.org/abs/1710.09412)
    """

    def __init__(self, alpha: float, num_classes: Optional[int] = None) -> None:
        """
        Args:
            alpha: the hyperparameter of Beta distribution used to sample mixup
            coefficient.
            num_classes: number of classes in the dataset.
        """
        self.alpha = alpha
        self.num_classes = num_classes

    def on_train_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Dict[str, Any],
        batch_idx: int,
        unused: Optional[int] = None,
    ) -> None:
        if batch["target"].ndim == 1:
            assert (
                self.num_classes is not None
            ), f"num_classes is expected for 1D target: {batch['target']}"
            batch["target"] = convert_to_one_hot(
                batch["target"].view(-1, 1), none_throws(self.num_classes)
            )
        else:
            assert batch["target"].ndim == 2, "target tensor shape must be 1D or 2D"

        c = (
            Beta(self.alpha, self.alpha)
            .sample(sample_shape=torch.Size())
            .to(device=batch["target"].device)
        )
        permuted_indices = torch.randperm(batch["target"].shape[0])
        for key in ["input", "target"]:
            batch[key] = c * batch[key] + (1.0 - c) * batch[key][permuted_indices, :]


@dataclass
class MixupTransformConf:
    _target_: str = get_class_name_str(MixupTransform)
    # pyre-fixme[4]: Attribute annotation cannot be `Any`.
    alpha: Any = MISSING
    num_classes: Optional[int] = None


cs: ConfigStore = ConfigStore.instance()
cs.store(
    group="callbacks/mixup_transform",
    name="mixup_transform",
    node=MixupTransformConf,
    package="mixup_transform",
)

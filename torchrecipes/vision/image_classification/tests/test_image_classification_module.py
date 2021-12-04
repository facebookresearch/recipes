# Copyright (c) Meta Platforms, Inc. and its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


# pyre-strict
from functools import partial

import testslide
import torch
from torchrecipes.vision.image_classification.module.image_classification import (
    ImageClassificationModule,
)
from torchvision.models.resnet import resnet18


class TestImageClassificationModule(testslide.TestCase):
    def test_custom_norm_weight_decay(self) -> None:
        module = ImageClassificationModule(
            model=resnet18(),
            loss=torch.nn.CrossEntropyLoss(),
            optim_fn=partial(torch.optim.SGD, lr=0.1),
            metrics={},
            norm_weight_decay=0.1,
        )

        param_groups = module.get_optimizer_param_groups()
        self.assertEqual(2, len(param_groups))
        self.assertEqual(0.1, param_groups[1]["weight_decay"])

    def test_custom_optimizer_interval(self) -> None:
        module = ImageClassificationModule(
            model=resnet18(),
            loss=torch.nn.CrossEntropyLoss(),
            optim_fn=partial(torch.optim.SGD, lr=0.1),
            lr_scheduler_fn=partial(torch.optim.lr_scheduler.StepLR, step_size=10),
            metrics={},
            lr_scheduler_interval="step",
        )
        optim = module.configure_optimizers()
        # pyre-ignore[16]: optim["lr_scheduler"] has key "interval"
        self.assertEqual("step", optim["lr_scheduler"]["interval"])

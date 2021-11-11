#!/usr/bin/env python3
from typing import Callable, Optional

import testslide
import torch
from torch import nn
from torchrecipes.vision.core.ops.fine_tuning_wrapper import FineTuningWrapper
from torchvision.models.resnet import resnet18
from torchvision.ops.misc import FrozenBatchNorm2d


class TestFineTuningWrapper(testslide.TestCase):
    def _get_model(
        self, freeze_trunk: bool, norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> FineTuningWrapper:
        trunk = resnet18(norm_layer=norm_layer)
        head = nn.Linear(in_features=512, out_features=2)
        return FineTuningWrapper(trunk, "flatten", head, freeze_trunk)

    def test_extraction(self) -> None:
        model = self._get_model(freeze_trunk=True, norm_layer=FrozenBatchNorm2d)
        inp = torch.randn(1, 3, 224, 224)
        out = model(inp)
        self.assertEqual(out.shape, torch.Size([1, 2]))

    def test_freeze_trunk(self) -> None:
        model = self._get_model(freeze_trunk=True, norm_layer=FrozenBatchNorm2d)
        # trunk should be frozon
        params = [x for x in model.trunk.parameters() if x.requires_grad]
        self.assertEqual(0, len(params))

        # head should be trainable
        params = [x for x in model.head.parameters() if x.requires_grad]
        self.assertEqual(2, len(params))

    def test_full_fine_tuning(self) -> None:
        model = self._get_model(freeze_trunk=False)
        params = [x for x in model.parameters() if x.requires_grad]
        self.assertEqual(len(list(model.parameters())), len(params))

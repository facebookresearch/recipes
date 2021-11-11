# pyre-strict

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
            optim_fn=torch.optim.SGD,
            metrics={},
            norm_weight_decay=0.1,
        )

        param_groups = module.get_optimizer_param_groups()
        self.assertEqual(2, len(param_groups))
        self.assertEqual(0.1, param_groups[1]["weight_decay"])

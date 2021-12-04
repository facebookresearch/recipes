# Copyright (c) Meta Platforms, Inc. and its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch
from torch import nn
from torch.fx.graph_module import GraphModule
from torchvision.models.feature_extraction import create_feature_extractor


class FineTuningWrapper(nn.Module):
    """
    A wrapper that creates the feature extractor from a pre-trained model
    and forward extracted features to layers to be fine-tuned.

    Args:
        trunk (nn.Module): model on which we will extract the features.
        feature_layer (str): the name of the node for which the activations
            will be returned.
        head (nn.Module): layers to be fine-tuned.
        freeze_trunk (bool): whether to freeze all parameters in the trunk.
            Default to True.
    """

    def __init__(
        self,
        trunk: nn.Module,
        feature_layer: str,
        head: nn.Module,
        freeze_trunk: bool = True,
    ) -> None:
        super().__init__()
        self.trunk: GraphModule = create_feature_extractor(trunk, [feature_layer])
        self.head = head
        self.feature_layer = feature_layer
        if freeze_trunk:
            self.freeze_trunk()

    def freeze_trunk(self) -> None:
        for param in self.trunk.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.trunk(x)
        return self.head(features[self.feature_layer])

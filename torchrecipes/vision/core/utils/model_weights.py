# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import logging
from typing import Dict

import torch
from iopath.common.file_io import g_pathmgr
from torch import nn

logger: logging.Logger = logging.getLogger(__name__)


def load_model_weights(
    module: nn.Module, weights_path: str, strict: bool = True
) -> nn.Module:
    """
    Loads model weights from given model weights path in-place.

    Args:
        module (nn.Module): module to be operated on.
        weights_path (str): path to model weight file in state dict format.
        strict (bool): whether to load state dict in strict mode.
    """
    with g_pathmgr.open(weights_path, "rb") as f:
        weights = torch.load(f, map_location="cpu")
        module.load_state_dict(weights, strict=strict)
        logger.info(f"Loaded model weights from {weights_path}.")
    return module


def extract_model_weights_from_checkpoint(
    checkpoint_path: str, model_name: str = ""
) -> Dict[str, torch.Tensor]:
    """
    Extracts model weights from given Lightning checkpoint.

    Args:
        checkpoint_path (str): path to the Lightning checkpoint.
        model_name (str): name of model attribute in the Lightning module.
            Set to empty if model is the Lightning module itself.
    """
    with g_pathmgr.open(checkpoint_path, "rb") as f:
        ckpt = torch.load(f, map_location="cpu")
        if "state_dict" not in ckpt:
            raise ValueError(
                'The checkpoint doesn\'t have key "state_dict",'
                " please make sure it's a valid Lightning checkpoint."
            )
        state_dict = ckpt["state_dict"]
        logger.info(f"Loaded state dict from checkpoint {checkpoint_path}.")

    prefix_len = 0 if not model_name else len(model_name) + 1  # e.g. "model."
    weights: Dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        if model_name and k.startswith(model_name):
            weights[k[prefix_len:]] = v
    if not weights:
        raise ValueError(
            f"No model weights found with prefix '{model_name}' in provided state dict"
        )
    return weights

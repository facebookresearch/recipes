# Copyright (c) Meta Platforms, Inc. and its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


"""
If you change the symbols exported in this file, you very likely want to change
the symbols expoerted in ./fb/_internal_patches.py as well.

For any buck-based build internally (within fbcode), this file is silently
replaced with the file located in ./fb/_internal_patches.py. This is to enable
use to silenetly swap out symbols (such as Checkpoint, Logger, etc.) between
internal-only implementations and external versions w/o requiring user involvement.
"""
from functools import wraps
from typing import Any

from pytorch_lightning.callbacks import ModelCheckpoint as OSSModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger as OSSTensorboardLogger

# Use tuples since these are not mutable.
_FB_ONLY_ARGS = (
    "has_user_data",
    "ttl_days",
    "manifold_bucket",
    "manifold_path",
    "num_retries",
    "save_torchscript",
    "save_quantized",
    "api_key",
    "enable_alert_service",
    "alert_configuration_path",
)


@wraps(OSSModelCheckpoint, updated=())
def ModelCheckpoint(**kwargs: Any) -> OSSModelCheckpoint:
    for arg_name in _FB_ONLY_ARGS:
        kwargs.pop(arg_name, None)
    return OSSModelCheckpoint(**kwargs)


@wraps(OSSTensorboardLogger, updated=())
def TensorBoardLogger(**kwargs: Any) -> OSSTensorboardLogger:
    for arg_name in _FB_ONLY_ARGS:
        kwargs.pop(arg_name, None)
    return OSSTensorboardLogger(**kwargs)


def log_run(**kwargs: Any) -> None:
    """Log Run."""
    pass

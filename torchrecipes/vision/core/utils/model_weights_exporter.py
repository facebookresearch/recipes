# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


#!/usr/bin/env python3
import argparse
import logging
from typing import Dict

import torch
from iopath.common.file_io import g_pathmgr
from torchrecipes.vision.core.utils.model_weights import (
    extract_model_weights_from_checkpoint,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        required=True,
        help="Path to the Lightning checkpoint.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="",
        help="Name of model attribute in Lightning module.",
    )
    parser.add_argument(
        "--model-weights-path", type=str, help="Export model weights to path."
    )

    args: argparse.Namespace = parser.parse_args()
    weights: Dict[str, torch.Tensor] = extract_model_weights_from_checkpoint(
        args.checkpoint_path, args.model_name
    )

    if args.model_weights_path:
        with g_pathmgr.open(args.model_weights_path, "wb") as f:
            torch.save(weights, f)
            logging.info(f"Saved model weights to {args.model_weights_path}.")


if __name__ == "__main__":
    main()  # pragma: no cover

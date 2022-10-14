#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import argparse

import fsspec
import torch

parser = argparse.ArgumentParser(description="Quantize the model from a snapshot")
parser.add_argument(
    "-i",
    "--input_path",
    type=str,
    required=True,
    help="Snapshot path to load. It can be a local path, S3 or Google Cloud Storage URL",
)
parser.add_argument(
    "-o",
    "--output_path",
    type=str,
    required=True,
    help="Snapshot path to save. It can be a local path, S3 or Google Cloud Storage URL",
)
parser.add_argument("-q", "--quantize", help="quantize the model", action="store_true")
parser.add_argument(
    "-t", "--torchscript", help="torchscript the model", action="store_true"
)


def main() -> None:
    args = parser.parse_args()

    fs, intput_path = fsspec.core.url_to_fs(args.input_path)
    with fs.open(intput_path, "rb") as f:
        model = torch.load(f, map_location="cpu")

    # quantize the model. Note that dynamic Quantization currently only
    # supports nn.Linear and nn.LSTM in qconfig_spec
    if args.quantize:
        print("quantizing the model...")
        model = torch.quantization.quantize_dynamic(
            model, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8
        )

    fs, output_path = fsspec.core.url_to_fs(args.output_path)
    with fs.open(output_path, "wb") as f:
        if args.torchscript:
            print("torchscripting the model...")
            model_jit = torch.jit.script(model)
            torch.jit.save(model_jit, f)
        else:
            torch.save(model, f)

    print(f"exported the module to {args.output_path}")


if __name__ == "__main__":
    main()

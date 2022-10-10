#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import argparse

import torch

from model import GPT, GPTConfig
from utils import get_filesystem

parser = argparse.ArgumentParser(description="Quantize the model from a snapshot")
parser.add_argument("-i", "--input_path", type=str, required=True,
                    help="Snapshot path to load. It can be a local path, S3 or Google Cloud Storage URL")
parser.add_argument("-o", "--output_path", type=str, required=True,
                    help="Snapshot path to save. It can be a local path, S3 or Google Cloud Storage URL")
parser.add_argument("-v", "--vocab_size", type=int, default=65, help="vocab size for the model")
parser.add_argument("-b", "--block_size", type=int, default=128, help="block size for the model")
parser.add_argument("-l", "--n_layer", type=int, default=2, help="number of layers for the model")
parser.add_argument("-d", "--n_head", type=int, default=2, help="number of heads for the model")
parser.add_argument("-e", "--n_embd", type=int, default=32, help="size of embedding for the model")


def main() -> None:
    args = parser.parse_args()

    mconf = GPTConfig(
        vocab_size=args.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
    )
    model = GPT(mconf)

    fs = get_filesystem(args.input_path)
    with fs.open(args.input_path, "rb") as f:
        model.load_state_dict(torch.load(f, map_location="cuda:0"))

    # quantize the model. Note that dynamic Quantization currently only 
    # supports nn.Linear and nn.LSTM in qconfig_spec
    model_quantized = torch.quantization.quantize_dynamic(
        model, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8
    )

    fs = get_filesystem(args.output_path)
    with fs.open(args.output_path, "wb") as f:
        torch.save(model_quantized, f)
    print(f"exported model to {args.output_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
CombinedModule includes model and its corresponding transform.
It's mainly used for inference from raw inputs, which will be converted
to tensors by transform then pass to model.
"""
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F


class CombinedModule(nn.Module):
    device: str

    def __init__(self, transform: nn.Module, model: nn.Module) -> None:
        super().__init__()
        self.transform = transform
        self.model = model
        self.device = ""

    def forward(self, text: str) -> str:
        tokens = self.transform(text)
        tokens = tokens.unsqueeze(0).to(self.device)
        generated_ids = self.generate(tokens).squeeze()
        return self.transform.decode(generated_ids)

    @torch.jit.export
    def set_device(self, device: str) -> None:
        self.device = device

    def top_k_logits(self, logits: Tensor, k: int) -> Tensor:
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[:, [-1]]] = -float("Inf")
        return out

    def generate(
        self,
        x: Tensor,
        steps: int = 100,
        temperature: float = 1.0,
        sample: bool = True,
        top_k: Optional[int] = 10,
    ) -> Tensor:
        """
        take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
        the sequence, feeding the predictions back into the model each time. Clearly the sampling
        has quadratic complexity unlike an RNN that is only linear, and has a finite context window
        of block_size, unlike an RNN that has an infinite context window.
        """
        block_size = 128
        for _ in range(steps):
            x_cond = (
                x if x.size(1) <= block_size else x[:, -block_size:]
            )  # crop context if needed
            logits = self.model(x_cond)
            # pluck the logits at the final step and scale by temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop probabilities to only the top k options
            if top_k is not None:
                logits = self.top_k_logits(logits, top_k)
            # apply softmax to convert to probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution or take the most likely
            if sample:
                ix = torch.multinomial(probs, num_samples=1)
            else:
                _, ix = torch.topk(probs, k=1, dim=-1)
            # append to the sequence and continue
            x = torch.cat((x, ix), dim=1)

        return x

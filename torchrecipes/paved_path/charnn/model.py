#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
The model taken from https://github.com/karpathy/minGPT
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""

import logging
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
)
from torch.distributed.fsdp.wrap import wrap

logger = logging.getLogger(__name__)


@dataclass
class OptimizerConfig:
    lr: float = 3e-4
    weight_decay: float = 0.1


def module_wrapper(module, fsdp=False, activation="noop") -> nn.Module:
    if not fsdp:
        return module

    if activation == "noop":
        return wrap(module)
    elif activation == "checkpoint":
        return wrap(checkpoint_wrapper(module))
    elif activation == "offload":
        return wrap(checkpoint_wrapper(module, offload_to_cpu=True))
    else:
        raise ValueError(f"Unrecognized activation mode {activation}")


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int
    n_layer: int
    n_head: int
    n_embd: int
    embd_pdrop: float = 0.1
    resid_pdrop: float = 0.1
    attn_pdrop: float = 0.1


class EmbeddingStem(nn.Module):
    def __init__(self, config: GPTConfig, dtype=torch.float32) -> None:
        super().__init__()

        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd, dtype=dtype)
        self.pos_emb = nn.Parameter(
            torch.zeros(1, config.block_size, config.n_embd, dtype=dtype)
        )
        self.drop = nn.Dropout(config.embd_pdrop)
        self.block_size = config.block_size

    def reset_parameters(self) -> None:
        self.tok_emb.reset_parameters()

    def forward(self, idx: Tensor) -> Tensor:
        b, t = idx.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        token_embeddings = self.tok_emb(idx)  # each index maps to a (learnable) vector
        position_embeddings = self.pos_emb[
            :, :t, :
        ]  # each position maps to a (learnable) vector
        return self.drop(token_embeddings + position_embeddings)


class MultiheadAttentionLayer(nn.Module):
    """
    A multi-head masked self-attention layer with a projection at the end.
    """

    def __init__(self, config, dtype=torch.float32) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        self.proj = nn.Linear(config.n_embd, config.n_embd, dtype=dtype)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )
        self.attn = torch.nn.MultiheadAttention(
            embed_dim=config.n_embd,
            num_heads=config.n_head,
            dropout=config.attn_pdrop,
            batch_first=True,
            dtype=dtype,
        )

    def forward(self, x) -> Tensor:
        _, seq_size, _ = x.size()
        y = self.attn(x, x, x, attn_mask=self.mask[0, 0, :seq_size, :seq_size])[0]
        y = self.resid_drop(self.proj(y))
        return y


class Block(nn.Module):
    """an unassuming Transformer block"""

    def __init__(self, config) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = MultiheadAttentionLayer(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x) -> Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    """the full GPT language model, with a context size of block_size"""

    def __init__(self, config) -> None:
        super().__init__()

        # input embedding stem
        self.emb_stem = EmbeddingStem(config)
        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.block_size = config.block_size
        self.apply(self._init_weights)
        rank = int(os.getenv("RANK", "0"))

        if rank == 0:
            print(
                "GPT Model Number of parameters: ",
                sum(p.numel() for p in self.parameters()),
            )

    def get_block_size(self) -> int:
        return self.block_size

    def _init_weights(self, module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, idx: Tensor) -> Tensor:
        _, t = idx.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        # forward the GPT model
        x = self.emb_stem(idx)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

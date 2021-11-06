#!/usr/bin/env python3

from typing import Optional

import pytorch_lightning as pl
from torchrecipes._internal_patches import ModelCheckpoint


def setup_checkpointing(
    model: pl.LightningModule,
    checkpoint_output_path: Optional[str] = None,
    load_path: Optional[str] = None,
) -> Optional[ModelCheckpoint]:
    checkpoint = (
        ModelCheckpoint(
            has_user_data=False,
            ttl_days=1,
            dirpath=checkpoint_output_path,
        )
        if checkpoint_output_path is not None
        else None
    )

    if load_path:
        print(f"loading checkpoint: {load_path}...")
        model.load_from_checkpoint(checkpoint_path=load_path)

    return checkpoint

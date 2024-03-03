# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


#!/usr/bin/env python3

import argparse
import os
import sys
from typing import List

import pytorch_lightning as pl
import torch
from pyre_extensions import none_throws
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from torchrec import EmbeddingBagCollection
from torchrec.datasets.criteo import DEFAULT_CAT_NAMES, DEFAULT_INT_NAMES
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrecipes.rec.accelerators.torchrec import TorchrecStrategy
from torchrecipes.rec.datamodules.criteo_datamodule import CriteoDataModule
from torchrecipes.rec.datamodules.random_rec_datamodule import RandomRecDataModule
from torchrecipes.rec.modules.lightning_dlrm import LightningDLRM


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="torchrec + lightning app")
    parser.add_argument(
        "--epochs", type=int, default=3, help="number of epochs to train"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="batch size to use for training"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="criteo_1t",
        help="dataset for experiment, current support criteo_1tb, criteo_kaggle",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        help="the path of the dataset",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="number of dataloader workers",
    )
    parser.add_argument(
        "--limit_train_batches",
        type=int,
        default=100,
        help="number of train batches",
    )
    parser.add_argument(
        "--limit_val_batches",
        type=int,
        default=100,
        help="number of val batches",
    )
    parser.add_argument(
        "--limit_test_batches",
        type=int,
        default=100,
        help="number of test batches",
    )
    parser.add_argument(
        "--num_embeddings",
        type=int,
        default=100_000,
        help="max_ind_size. The number of embeddings in each embedding table. Defaults"
        " to 100_000 if num_embeddings_per_feature is not supplied.",
    )
    parser.add_argument(
        "--num_embeddings_per_feature",
        type=str,
        default=None,
        help="Comma separated max_ind_size per sparse feature. The number of embeddings"
        " in each embedding table. 26 values are expected for the Criteo dataset.",
    )
    parser.add_argument(
        "--dense_arch_layer_sizes",
        type=str,
        default="512,256,64",
        help="Comma separated layer sizes for dense arch.",
    )
    parser.add_argument(
        "--over_arch_layer_sizes",
        type=str,
        default="512,512,256,1",
        help="Comma separated layer sizes for over arch.",
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=64,
        help="Size of each embedding.",
    )
    parser.add_argument(
        "--tensorboard_save_dir",
        type=str,
        help="Path to save tensorboard.",
        required=False,
    )
    parser.add_argument("--load_path", type=str, help="Checkpoint path to load from.")
    parser.add_argument(
        "--checkpoint_output_path",
        type=str,
        help="Path to place checkpoints.",
        required=False,
    )
    parser.add_argument(
        "--disable_checkpointing",
        dest="checkpointing",
        action="store_false",
        help="Disable checkpointing.",
    )
    parser.add_argument(
        "--undersampling_rate",
        type=float,
        help="Desired proportion of zero-labeled samples to retain (i.e. undersampling zero-labeled rows)."
        " Ex. 0.3 indicates only 30pct of the rows with label 0 will be kept."
        " All rows with label 1 will be kept. Value should be between 0 and 1."
        " When not supplied, no undersampling occurs.",
    )
    parser.add_argument(
        "--seed",
        type=float,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--pin_memory",
        dest="pin_memory",
        action="store_true",
        help="Use pinned memory when loading data.",
    )
    parser.add_argument(
        "--random_dataloader",
        dest="random_dataloader",
        action="store_true",
        help="Use random data loader.",
    )
    parser.set_defaults(pin_memory=False)
    parser.set_defaults(random_dataloader=True)
    parser.set_defaults(checkpointing=True)
    return parser.parse_args(argv)


def main(argv: List[str]) -> None:
    args = parse_args(argv)

    random_dataloader = args.random_dataloader

    if args.num_embeddings_per_feature is not None:
        num_embeddings_per_feature = list(
            map(int, args.num_embeddings_per_feature.split(","))
        )
        num_embeddings = None
    else:
        num_embeddings_per_feature = None
        num_embeddings = args.num_embeddings

    if random_dataloader:
        datamodule = RandomRecDataModule(
            keys=DEFAULT_CAT_NAMES,
            hash_size=num_embeddings,
            hash_sizes=num_embeddings_per_feature,
            batch_size=args.batch_size,
            ids_per_feature=1,
            pin_memory=args.pin_memory,
            num_workers=args.num_workers,
            num_dense=len(DEFAULT_INT_NAMES),
            manual_seed=args.seed,
        )
    elif args.dataset_name in ["criteo_1t", "criteo_kaggle"]:
        datamodule = CriteoDataModule(
            dataset_name=args.dataset_name,
            num_days=1,
            batch_size=args.batch_size,
            num_days_test=1,
            num_workers=args.num_workers,
            undersampling_rate=args.undersampling_rate,
            seed=args.seed,
            num_embeddings=num_embeddings,
            num_embeddings_per_feature=num_embeddings_per_feature,
            pin_memory=args.pin_memory,
            dataset_path=args.dataset_path,
        )
    else:
        raise ValueError(
            f"Unknown dataset {args.dataset_name}. "
            + "Please choose {criteo_1t, criteo_kaggle} for dataset_name"
        )
    keys = datamodule.keys
    embedding_dim = args.embedding_dim

    eb_configs = [
        EmbeddingBagConfig(
            name=f"t_{feature_name}",
            embedding_dim=embedding_dim,
            num_embeddings=(
                none_throws(num_embeddings_per_feature)[feature_idx]
                if num_embeddings is None
                else num_embeddings
            ),
            feature_names=[feature_name],
        )
        for feature_idx, feature_name in enumerate(keys)
    ]
    sharded_module_kwargs = {}
    if args.over_arch_layer_sizes is not None:
        sharded_module_kwargs["over_arch_layer_sizes"] = list(
            map(int, args.over_arch_layer_sizes.split(","))
        )

    sharded_model = LightningDLRM(
        EmbeddingBagCollection(tables=eb_configs, device=torch.device("meta")),
        batch_size=args.batch_size,
        dense_in_features=len(DEFAULT_INT_NAMES),
        dense_arch_layer_sizes=list(map(int, args.dense_arch_layer_sizes.split(","))),
        over_arch_layer_sizes=list(map(int, args.over_arch_layer_sizes.split(","))),
    )

    checkpoint = ModelCheckpoint(dirpath=args.checkpoint_output_path)
    callbacks = [checkpoint] if checkpoint is not None else []

    logger = (
        TensorBoardLogger(save_dir=args.tensorboard_save_dir)
        if args.tensorboard_save_dir is not None
        else False
    )

    trainer = pl.Trainer(
        logger=logger,
        max_epochs=args.epochs,
        enable_checkpointing=args.checkpointing,
        # pyre-fixme[6]: For 4th param expected `Union[None, List[Callback],
        #  Callback]` but got `Union[List[typing.Any], List[ModelCheckpoint]]`.
        callbacks=callbacks,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        limit_test_batches=args.limit_test_batches,
        strategy=TorchrecStrategy(),
        accelerator=("gpu" if torch.cuda.is_available() else "cpu"),
        devices=os.environ.get("LOCAL_WORLD_SIZE", 1),
    )

    trainer.fit(sharded_model, datamodule=datamodule)


def invoke_main() -> None:
    main(sys.argv[1:])


if __name__ == "__main__":
    invoke_main()  # pragma: no cover

#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import logging
import os
import random
import socket
import uuid
from typing import Tuple

import fsspec
import hydra
import torch
import torch.distributed as dist
import torch.nn as nn
import torchsnapshot
from char_dataset import CharDataset
from combined_module import CombinedModule
from model import GPT, GPTConfig, OptimizerConfig
from omegaconf import DictConfig
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import random_split
from trainer import Trainer, TrainerConfig
from utils import get_realpath

logger = logging.getLogger(__name__)


def get_fq_hostname() -> str:
    return socket.getfqdn(socket.gethostname())


def set_env() -> None:
    os.environ["RANK"] = os.environ.get("RANK", "0")
    os.environ["WORLD_SIZE"] = os.environ.get("WORLD_SIZE", "1")
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29830")
    os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
    os.environ["LOCAL_RANK"] = os.environ.get("LOCAL_RANK", "0")
    os.environ["TORCHELASTIC_RUN_ID"] = os.environ.get(
        "TORCHELASTIC_RUN_ID", str(uuid.uuid4()).split("-")[0]
    )


def get_job_name() -> str:
    uid = os.environ["TORCHELASTIC_RUN_ID"]
    return f"run-{uid}"


def get_device() -> torch.device:
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    return device


def get_ddp_model_and_optimizer(
    gpt_config: GPTConfig, opt_config: OptimizerConfig
) -> Tuple[torch.nn.Module, torch.optim.Optimizer]:
    # Create new GPT Model on CPU
    model = GPT(gpt_config)
    device = get_device()
    device_ids = None
    if device.type == "cuda":
        model = model.to(device)
        device_ids = [device.index]
    model = DistributedDataParallel(
        model,
        device_ids=device_ids,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=opt_config.lr, weight_decay=opt_config.weight_decay
    )
    return model, optimizer


def get_model_and_optimizer(
    type: str,
    gpt_config: GPTConfig,
    opt_config: OptimizerConfig,
) -> Tuple[torch.nn.Module, torch.optim.Optimizer]:
    if type == "ddp":
        return get_ddp_model_and_optimizer(gpt_config, opt_config)
    raise RuntimeError(f"Unknown type: {type}. Allowed values: [ddp]")


def setup_process_group() -> None:
    device = get_device()
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if device.type == "cuda":
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    else:
        dist.init_process_group("gloo", rank=rank, world_size=world_size)


def generate_seq(
    cfg: DictConfig, module: torch.nn.Module, dataset: CharDataset
) -> None:
    if dist.get_rank() == 0:
        context = cfg["charnn"]["phrase"]
        completion = module(context)
        print(completion)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def save_module(transform: nn.Module, model: nn.Module, save_path: str) -> None:
    module = CombinedModule(transform=transform, model=model)
    module.eval()

    fs, path = fsspec.core.url_to_fs(save_path)
    dirname = os.path.dirname(path)
    if not fs.exists(dirname):
        fs.mkdirs(dirname)
    logger.info(f"Saving CombinedModule to {save_path}")
    with fs.open(path, "wb") as f:
        torch.save(module, f)


@hydra.main(config_path=".", config_name="trainer_config")
def main(cfg: DictConfig) -> None:
    # Setup distributed for distributed training
    # TODO: @stevenliu clean up. As we use torchx, most of them are not required
    set_env()
    setup_process_group()

    device = get_device()
    set_seed(42)

    train_cfg = cfg["trainer"]
    job_name = train_cfg["job_name"] if train_cfg.get("job_name") else get_job_name()

    # Data Loading
    data_path = get_realpath(cfg["dataset"]["path"])
    logger.info(
        f"{get_fq_hostname()}:{os.getpid()}:{device} Running charNN {job_name}, data_path: {data_path}"
    )
    block_size = 128  # spatial extent of the model for its context
    dataset = CharDataset(data_path, block_size)
    data_len = len(dataset)
    train_len = int(data_len * 0.9)
    train_dataset, test_dataset = random_split(
        dataset, [train_len, data_len - train_len]
    )

    mconf = GPTConfig(
        vocab_size=dataset.vocab_size,
        block_size=dataset.block_size,
        n_layer=cfg["model"]["n_layer"],
        n_head=cfg["model"]["n_head"],
        n_embd=cfg["model"]["n_embd"],
    )

    train_cfg["work_dir"] = os.path.join(train_cfg.get("work_dir", ""), job_name)
    tconf = TrainerConfig(
        work_dir=train_cfg["work_dir"],
        job_name=job_name,
        max_epochs=train_cfg["max_epochs"],
        batch_size=train_cfg["batch_size"],
        data_loader_workers=train_cfg["data_loader_workers"],
        enable_profile=train_cfg["enable_profile"],
        # TODO: @stevenliu remove log_dir. infer it from work_dir
        log_dir=os.path.join(train_cfg["work_dir"], "logs"),
    )
    opt_conf = OptimizerConfig(
        lr=cfg["opt"]["lr"], weight_decay=cfg["opt"]["weight_decay"]
    )
    model, optimizer = get_model_and_optimizer(cfg["charnn"]["dist"], mconf, opt_conf)

    # app_state will be saved or restored for checkpointing
    progress = torchsnapshot.StateDict(current_epoch=0)
    app_state = {
        "model": model,
        "optimizer": optimizer,
        "progress": progress,
    }

    if train_cfg.get("snapshot_path", None):
        snapshot = torchsnapshot.Snapshot(train_cfg["snapshot_path"])
        print(f"Restoring snapshot from path: {train_cfg['snapshot_path']}")
        snapshot.restore(app_state=app_state)

    if cfg["charnn"]["task"] == "train":
        # Model Training
        trainer = Trainer(
            model,
            optimizer,
            train_dataset,
            test_dataset,
            tconf,
            device,
            progress["current_epoch"],
        )
        trainer.fit(app_state, max_iter=cfg.get("max_iter", -1))

        # Save model and its transform together
        save_module(
            transform=dataset.transform,
            model=model.module,  # save the vanilla model instead of DDP wrapped model
            save_path=os.path.join(train_cfg["work_dir"], "modules/last.pt"),
        )
    elif cfg["charnn"]["task"] == "generate":
        module = CombinedModule(transform=dataset.transform, model=model)
        module.set_device(device)
        generate_seq(cfg, module, train_dataset.dataset)
    else:
        raise RuntimeError(f"Unknown task: {cfg['charnn']['task']}")


if __name__ == "__main__":
    main()

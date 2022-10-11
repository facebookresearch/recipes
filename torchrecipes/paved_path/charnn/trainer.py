#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import logging
import os
from dataclasses import dataclass
from typing import Dict, Optional

import fsspec
import torch
import torch.optim as optim
from torch.nn import Module
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torchsnapshot import Snapshot, Stateful

logger = logging.getLogger(__name__)


@dataclass
class TrainerConfig:
    work_dir: str
    job_name: str
    max_epochs: int = 10
    batch_size: int = 64
    data_loader_workers: int = 0
    enable_profile: bool = False
    log_dir: Optional[str] = None


def get_raw_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: optim.Optimizer,
        train_dataset: Dataset,
        test_dataset: Dataset,
        config: TrainerConfig,
        device: Optional[torch.device] = None,
        start_epoch: int = 0,
    ):
        self.model = model
        self.optimizer = optimizer
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.start_epoch = start_epoch

        self.device = device
        self.rank = int(os.environ["RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])
        self.tb_writer = self._get_tb_writer()

    def _get_log_dir(self) -> str:
        return f"{self.config.log_dir}/{self.config.job_name}/{self.rank}"

    def _get_tb_writer(self) -> Optional[SummaryWriter]:
        if self.config.log_dir:
            return SummaryWriter(log_dir=self._get_log_dir())
        else:
            return None

    def _try_create_profiler(self) -> Optional[torch.profiler.profile]:
        if not self.config.enable_profile:
            return None
        if self.config.log_dir is None:
            raise RuntimeError(
                "In order to use profiling pass the log dir `+trainer.log_dir=#YOUR_LOG_DIR`"
            )
        return torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=5),
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                self._get_log_dir()
            ),
        )

    def run_batch(self, x, y, train: bool = True) -> float:
        with torch.set_grad_enabled(train):
            _, loss = self.model(x, y)

        if train:
            self.model.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

        return loss.item()

    def run_epoch(self, epoch: int, max_iter: int = -1) -> None:
        train_sampler = DistributedSampler(
            self.train_dataset,
            rank=self.rank,
            num_replicas=self.world_size,
            shuffle=True,
        )
        train_loader = DataLoader(
            self.train_dataset,
            pin_memory=True,
            batch_size=self.config.batch_size,
            num_workers=self.config.data_loader_workers,
            sampler=train_sampler,
        )

        test_loader = DataLoader(
            self.test_dataset,
            pin_memory=True,
            batch_size=self.config.batch_size,
            num_workers=self.config.data_loader_workers,
        )

        prof = self._try_create_profiler()
        try:
            self.model.train()
            for it, (x, y) in enumerate(train_loader):
                x = x.to(self.device)
                y = y.to(self.device)
                train_batch_loss = self.run_batch(x, y, train=True)
                if prof:
                    prof.step()
                if self.tb_writer:
                    self.tb_writer.add_scalar(
                        f"train_loss_{epoch}", train_batch_loss, it
                    )
                if it % 100 == 0:
                    print(
                        f"{self.rank}: epoch {epoch} iter {it}: train loss {train_batch_loss:.5f}"
                    )
                if max_iter > 0 and it >= max_iter:
                    break
            self.model.eval()
            for it, (x, y) in enumerate(test_loader):
                x = x.to(self.device)
                y = y.to(self.device)
                test_batch_loss = self.run_batch(x, y, train=False)
                if self.tb_writer:
                    self.tb_writer.add_scalar(f"test_loss_{epoch}", test_batch_loss, it)
                if it % 100 == 0:
                    print(
                        f"{self.rank}: epoch {epoch} iter {it}: test loss {test_batch_loss:.5f}"
                    )
                if max_iter > 0 and it >= max_iter:
                    break

        finally:
            if prof:
                prof.stop()
            if self.tb_writer:
                self.tb_writer.flush()

    def export(self, model: Module, path: str) -> None:
        fs, p = fsspec.core.url_to_fs(path)
        dirname = os.path.dirname(p)
        if not fs.exists(dirname):
            fs.mkdirs(dirname)

        logger.info(f"Exporting model to {path}")
        model.eval()
        with fs.open(path, "wb") as f:
            torch.save(model.state_dict(), f)

    def fit(self, app_state: Dict[str, Stateful], max_iter: int = -1) -> None:
        snapshot_path = ""
        progress = app_state["progress"]
        for epoch in range(progress["current_epoch"], self.config.max_epochs):
            self.run_epoch(epoch, max_iter)
            progress["current_epoch"] += 1

            # save a snapshot per epoch
            if epoch == self.config.max_epochs - 1:
                snapshot_path = os.path.join(self.config.work_dir, "snapshots/last")
            else:
                snapshot_path = os.path.join(
                    self.config.work_dir, f"snapshots/epoch-{progress['current_epoch']}"
                )
            snapshot = Snapshot.take(path=snapshot_path, app_state=app_state)
            logger.info(f"Saving snapshot to {snapshot.path}")

        model_path = os.path.join(self.config.work_dir, "models/last.pt")
        self.export(app_state["model"].module, model_path)
        return snapshot_path

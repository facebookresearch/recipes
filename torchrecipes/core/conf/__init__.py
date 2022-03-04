# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Any, List, Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class TrainerConf:
    """
    Config for Lightning Trainer.
    See the usage of each arg in pytorch_lightning/trainer/trainer.py
    This same class is used for Hydra, therefore Hydra restrictions apply.

    Please do not use `Any` type with `None` as default value.
    FBLearner UI will not allow that. #@fb-only
    """

    accelerator: Optional[str] = None
    # Union[int, Dict[int, int], List[list]]
    # pyre-fixme[4]: Attribute annotation cannot be `Any`.
    accumulate_grad_batches: Any = None
    amp_backend: str = "native"
    amp_level: Optional[str] = None
    # Union[bool, str]
    # pyre-fixme[4]: Attribute annotation cannot be `Any`.
    auto_lr_find: Any = False
    # Union[bool, str]
    # pyre-fixme[4]: Attribute annotation cannot be `Any`.
    auto_scale_batch_size: Any = False
    auto_select_gpus: bool = False
    benchmark: bool = False
    check_val_every_n_epoch: int = 1
    checkpoint_callback: Optional[bool] = None
    default_root_dir: Optional[str] = None
    detect_anomaly: bool = False
    deterministic: bool = False
    devices: Optional[int] = None
    enable_checkpointing: bool = True
    enable_model_summary: bool = True
    enable_progress_bar: bool = True
    # Union[int, bool]
    # pyre-fixme[4]: Attribute annotation cannot be `Any`.
    fast_dev_run: Any = False
    flush_logs_every_n_steps: Optional[int] = None
    # Optional[Union[List[int], str, int]]
    # pyre-fixme[4]: Attribute annotation cannot be `Any`.
    gpus: Any = None
    # Optional[Union[int, float]]
    # pyre-fixme[4]: Attribute annotation cannot be `Any`.
    gradient_clip_val: Optional[Any] = None
    gradient_clip_algorithm: Optional[str] = None
    ipus: Optional[int] = None
    # Union[int, float]
    # pyre-fixme[4]: Attribute annotation cannot be `Any`.
    limit_train_batches: Any = 1.0
    # Union[int, float]
    # pyre-fixme[4]: Attribute annotation cannot be `Any`.
    limit_val_batches: Any = 1.0
    # Union[int, float]
    # pyre-fixme[4]: Attribute annotation cannot be `Any`.
    limit_test_batches: Any = 1.0
    log_gpu_memory: Optional[str] = None
    log_every_n_steps: int = 50
    # Union[bool, LightningLoggerBase]
    # pyre-fixme[4]: Missing attribute annotation
    logger: Any = True
    plugins: Optional[List[str]] = None
    prepare_data_per_node: Optional[bool] = None
    process_position: int = 0
    profiler: Optional[str] = None
    progress_bar_refresh_rate: Optional[int] = None
    # Union[int, float]
    # pyre-fixme[4]: Attribute annotation cannot be `Any`.
    overfit_batches: Any = 0.0
    # Union[float, str], precision can be "bf16"
    # pyre-fixme[4]: Attribute annotation cannot be `Any`.
    precision: Any = 32
    max_epochs: Optional[int] = None
    min_epochs: Optional[int] = None
    max_steps: int = -1
    min_steps: Optional[int] = None
    # Optional[Union[str, timedelta, Dict[str, int]]]
    # pyre-fixme[4]: Attribute annotation cannot be `Any`.
    max_time: Any = None
    move_metrics_to_cpu: bool = False
    multiple_trainloader_mode: str = "max_size_cycle"
    num_nodes: int = 1
    num_processes: Optional[int] = None
    num_sanity_val_steps: int = 2
    reload_dataloaders_every_n_epochs: int = 0
    replace_sampler_ddp: bool = False
    resume_from_checkpoint: Optional[str] = None
    stochastic_weight_avg: bool = False
    # Union[str, Strategy]
    # pyre-fixme[4]: Attribute annotation cannot be `Any`.
    strategy: Optional[Any] = None
    sync_batchnorm: bool = False
    terminate_on_nan: Optional[bool] = None
    tpu_cores: Optional[int] = None
    # Union[int, float, str]
    # pyre-fixme[4]: Attribute annotation cannot be `Any`.
    track_grad_norm: Any = -1
    # Union[int, float]
    # pyre-fixme[4]: Attribute annotation cannot be `Any`.
    val_check_interval: Any = 1.0
    weights_save_path: Optional[str] = None
    weights_summary: Optional[str] = "top"


@dataclass
class ModuleConf:
    pass


@dataclass
class DataModuleConf:
    pass


@dataclass
class TrainAppConf:
    module: ModuleConf = MISSING
    datamodule: Optional[DataModuleConf] = None
    trainer: TrainerConf = MISSING


# pyre-fixme[5]: Global expression must be annotated.
cs = ConfigStore.instance()
cs.store(group="schema/trainer", name="trainer", node=TrainerConf, package="trainer")

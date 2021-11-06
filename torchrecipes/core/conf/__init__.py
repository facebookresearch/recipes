# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

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
    checkpoint_callback: bool = True
    default_root_dir: Optional[str] = None
    deterministic: bool = False
    # Union[int, bool]
    # pyre-fixme[4]: Attribute annotation cannot be `Any`.
    fast_dev_run: Any = False
    flush_logs_every_n_steps: int = 100
    # Optional[Union[List[int], str, int]]
    # pyre-fixme[4]: Attribute annotation cannot be `Any`.
    gpus: Any = None
    gradient_clip_val: float = 0.0
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
    prepare_data_per_node: bool = True
    process_position: int = 0
    profiler: Optional[str] = None
    progress_bar_refresh_rate: int = 1
    # Union[int, float]
    # pyre-fixme[4]: Attribute annotation cannot be `Any`.
    overfit_batches: Any = 0.0
    # Union[float, str], precision can be "bf16"
    # pyre-fixme[4]: Attribute annotation cannot be `Any`.
    precision: Any = 32
    max_epochs: int = 1000
    min_epochs: int = 1
    max_steps: Optional[int] = None
    min_steps: Optional[int] = None
    move_metrics_to_cpu: bool = False
    num_nodes: int = 1
    num_processes: int = 1
    num_sanity_val_steps: int = 2
    reload_dataloaders_every_epoch: bool = False
    reload_dataloaders_every_n_epochs: int = 0
    replace_sampler_ddp: bool = False
    resume_from_checkpoint: Optional[str] = None
    strategy: Optional[str] = None
    sync_batchnorm: bool = False
    terminate_on_nan: bool = False
    tpu_cores: Optional[int] = None
    # Union[int, float, str]
    # pyre-fixme[4]: Attribute annotation cannot be `Any`.
    track_grad_norm: Any = -1
    # Union[int, float]
    # pyre-fixme[4]: Attribute annotation cannot be `Any`.
    val_check_interval: Any = 1.0
    weights_save_path: Optional[str] = None
    weights_summary: Optional[str] = "top"
    # Optional[Union[str, timedelta, Dict[str, int]]]
    # pyre-fixme[4]: Attribute annotation cannot be `Any`.
    max_time: Any = None
    detect_anomaly: bool = False


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

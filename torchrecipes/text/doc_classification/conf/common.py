# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Optional, List, Tuple

import torch
import torchtext
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from torchrecipes.text.doc_classification.transform.doc_classification_text_transform import (
    DocClassificationTextTransformConf,
)
from torchrecipes.utils.config_utils import get_class_name_str
from torchtext.datasets.sst2 import SST2
from torchtext.transforms import LabelToIndex


@dataclass
class DatasetConf:
    pass


@dataclass
class SST2DatasetConf(DatasetConf):
    _target_: str = get_class_name_str(SST2)
    root: str = MISSING


@dataclass
class LabelTransformConf:
    _target_: str = get_class_name_str(LabelToIndex)
    label_names: Optional[List[str]] = None
    label_path: Optional[str] = None
    sort_names: bool = False


@dataclass
class TransformConf:
    pass


@dataclass
class DocClassificationTransformConf(TransformConf):
    transform: DocClassificationTextTransformConf = MISSING
    label_transform: Optional[LabelTransformConf] = None
    num_labels: int = MISSING


@dataclass
class HeadConf:
    _target_: str = MISSING


@dataclass
class ClassificationHeadConf(HeadConf):
    _target_: str = get_class_name_str(torchtext.models.RobertaClassificationHead)
    num_classes: int = MISSING
    input_dim: int = MISSING
    inner_dim: Optional[int] = None
    dropout: float = 0


@dataclass
class XLMREncoderConf:
    _target_: str = get_class_name_str(torchtext.models.RobertaEncoderConf)
    vocab_size: int = 250002
    embedding_dim: int = 768
    ffn_dimension: int = 3072
    padding_idx: int = 1
    max_seq_len: int = 514
    num_attention_heads: int = 12
    num_encoder_layers: int = 12
    dropout: float = 0.1
    scaling: Optional[float] = None
    normalize_before: bool = False


@dataclass
class ModelConf:
    pass


@dataclass
class XLMRClassificationModelConf(ModelConf):
    _target_: str = "torchtext.models.RobertaBundle.build_model"
    encoder_conf: XLMREncoderConf = XLMREncoderConf()
    head: HeadConf = ClassificationHeadConf()
    freeze_encoder: bool = False
    checkpoint: Optional[
        str
    ] = "https://download.pytorch.org/models/text/xlmr.base.encoder.pt"


@dataclass
class OptimConf:
    pass


@dataclass
class AdamWConf(OptimConf):
    _target_: str = get_class_name_str(torch.optim.AdamW)
    lr: float = 1e-3
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0
    amsgrad: bool = False


@dataclass
class AdamConf(OptimConf):
    _target_: str = get_class_name_str(torch.optim.Adam)
    lr: float = 1e-3
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8


cs: ConfigStore = ConfigStore.instance()
cs.store(
    group="module/model",
    name="xlmrbase_classifier",
    node=XLMRClassificationModelConf,
)

cs.store(group="schema/datamodule/dataset", name="dataset", node=DatasetConf)
cs.store(group="datamodule/dataset", name="sst2_dataset", node=SST2DatasetConf)

cs.store(group="transform", name="label_transform", node=LabelTransformConf)
cs.store(group="transform", name="transform", node=TransformConf)

cs.store(group="schema/task/optim", name="optim", node=OptimConf)
cs.store(group="task/optim", name="adam", node=AdamConf)
cs.store(group="task/optim", name="adamw", node=AdamWConf)

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Optional, List

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


cs: ConfigStore = ConfigStore.instance()

cs.store(group="schema/datamodule/dataset", name="dataset", node=DatasetConf)
cs.store(group="datamodule/dataset", name="sst2_dataset", node=SST2DatasetConf)

cs.store(group="transform", name="label_transform", node=LabelTransformConf)
cs.store(group="transform", name="transform", node=TransformConf)

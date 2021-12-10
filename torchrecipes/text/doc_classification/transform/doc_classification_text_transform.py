# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from dataclasses import dataclass
from typing import Any, Dict, List

import torch
import torch.nn as nn
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from torchrecipes.utils.config_utils import get_class_name_str
from torchtext.functional import to_tensor
from torchtext.models.roberta.transforms import XLMRobertaModelTransform


class DocClassificationTextTransform(nn.Module):
    def __init__(
        self,
        vocab_path: str,
        spm_model_path: str,
        text_column: str = "text",
        token_ids_column: str = "token_ids",
        **kwargs: Any
    ) -> None:
        super().__init__()
        self.xlmr_roberta_model_transform = XLMRobertaModelTransform(
            vocab_path, spm_model_path, **kwargs
        )
        self.text_column = text_column
        self.token_ids_column = token_ids_column

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        texts = batch[self.text_column]
        assert torch.jit.isinstance(texts, List[str])

        tokens_list = self.xlmr_roberta_model_transform(texts)
        tokens_tensor: torch.Tensor = to_tensor(
            tokens_list, self.xlmr_roberta_model_transform.pad_idx
        )
        batch[self.token_ids_column] = tokens_tensor
        return batch


@dataclass
class DocClassificationTextTransformConf:
    _target_: str = get_class_name_str(DocClassificationTextTransform)
    vocab_path: str = MISSING
    spm_model_path: str = MISSING


# pyre-fixme[5]: Global expression must be annotated.
cs = ConfigStore.instance()
cs.store(
    group="transform",
    name="doc_classification_text_transform",
    node=DocClassificationTextTransformConf,
)

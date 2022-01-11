# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import os.path
from dataclasses import dataclass
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torchtext
import torchtext.transforms as T
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from torchrecipes.utils.config_utils import get_class_name_str
from torchtext.functional import to_tensor


class DocClassificationTextTransform(nn.Module):
    def __init__(
        self,
        vocab_path: str,
        spm_model_path: str,
        text_column: str = "text",
        token_ids_column: str = "token_ids",
        pad_idx: int = 1,
    ) -> None:
        super().__init__()

        if os.path.exists(vocab_path):
            vocab = torch.load(vocab_path)
        else:
            vocab = torchtext._download_hooks.load_state_dict_from_url(vocab_path)

        self.xlmr_roberta_model_transform = T.Sequential(
            T.SentencePieceTokenizer(spm_model_path),
            T.VocabTransform(vocab),
            T.Truncate(254),
            T.AddToken(token=0, begin=True),
            T.AddToken(token=2, begin=False),
        )
        self.text_column = text_column
        self.token_ids_column = token_ids_column
        self.pad_idx = pad_idx

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        texts = batch[self.text_column]
        assert torch.jit.isinstance(texts, List[str])

        tokens_list = self.xlmr_roberta_model_transform(texts)
        tokens_tensor: torch.Tensor = to_tensor(tokens_list, self.pad_idx)
        batch[self.token_ids_column] = tokens_tensor
        return batch


@dataclass
class DocClassificationTextTransformConf:
    _target_: str = get_class_name_str(DocClassificationTextTransform)
    vocab_path: str = MISSING
    spm_model_path: str = MISSING
    text_column: str = "text"
    token_ids_column: str = "token_ids"
    pad_idx: int = 1


# pyre-fixme[5]: Global expression must be annotated.
cs = ConfigStore.instance()
cs.store(
    group="transform",
    name="doc_classification_text_transform",
    node=DocClassificationTextTransformConf,
)

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import os.path
import shutil
from typing import Tuple, Union

from torchtext.datasets import sst2

_DATA_DIR_PATH: str = os.path.realpath(
    os.path.join(os.path.dirname(__file__), "..", "data")
)


def get_asset_path(*paths: Union[str, Tuple[str]]) -> str:
    """Return full path of a test asset"""
    return os.path.join(_DATA_DIR_PATH, *paths)


def copy_asset(cur_path: str, new_path: str) -> None:
    new_path_dir = os.path.dirname(new_path)
    if not os.path.exists(new_path_dir):
        os.makedirs(new_path_dir)
    shutil.copy(cur_path, new_path)


def copy_partial_sst2_dataset(root_dir: str) -> None:
    cur_path = get_asset_path(sst2.DATASET_NAME, sst2._PATH)
    new_path = os.path.join(root_dir, sst2.DATASET_NAME, sst2._PATH)
    copy_asset(cur_path, new_path)

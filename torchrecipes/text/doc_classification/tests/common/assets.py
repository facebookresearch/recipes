# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import os.path
import shutil
from typing import Tuple, Union

from torchtext.experimental.datasets import sst2

_DATA_DIR_PATH: str = os.path.realpath(
    os.path.join(os.path.dirname(__file__), "..", "data")
)


def get_asset_path(*paths: Union[str, Tuple[str]]) -> str:
    """Return full path of a test asset"""
    return os.path.join(_DATA_DIR_PATH, *paths)


def copy_partial_sst2_dataset(root_dir: str) -> None:
    asset_path = get_asset_path(sst2.DATASET_NAME, sst2._PATH)
    data_folder = os.path.join(root_dir, sst2.DATASET_NAME)
    data_path = os.path.join(data_folder, sst2._PATH)
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    shutil.copy(asset_path, data_path)

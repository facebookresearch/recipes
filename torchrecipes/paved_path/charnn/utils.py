#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os


def get_realpath(path: str) -> str:
    if "://" in path or os.path.isabs(path):
        return path

    work_dir = os.path.dirname(__file__)
    return os.path.join(work_dir, path)

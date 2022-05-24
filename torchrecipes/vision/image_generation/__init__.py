# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


#!/usr/bin/env python3


def register_components() -> None:
    """
    Calls register_components() for all subfolders so we can register
    subcomponents to Hydra's ConfigStore.
    """
    import torchrecipes.vision.image_generation.module.gan  # noqa
    import torchrecipes.vision.image_generation.module.infogan  # noqa
    import torchrecipes.vision.image_generation.train_app  # noqa

# Copyright (c) Meta Platforms, Inc. and its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


#!/usr/bin/env python3

import torchrecipes.core.conf  # noqa
import torchrecipes.vision.data.modules  # noqa

# Components to register with this config
from torchrecipes.vision.image_classification import register_components

register_components()

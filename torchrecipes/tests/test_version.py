# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import testslide


class VersionTest(testslide.TestCase):
    def test_can_get_version(self) -> None:
        import torchrecipes

        self.assertIsNotNone(torchrecipes.__version__)

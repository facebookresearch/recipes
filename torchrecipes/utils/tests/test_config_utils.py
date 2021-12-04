# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


# pyre-strict

import testslide
from torchrecipes.utils.config_utils import (
    config_entry,
    get_class_config_method,
    get_class_name_str,
)


class TestConfigUtils(testslide.TestCase):
    def test_annotation_success(self) -> None:
        class TestClass:
            @config_entry
            @staticmethod
            def from_config(config: object) -> "TestClass":
                return TestClass()

        self.assertEqual(
            get_class_name_str(TestClass),
            "torchrecipes.utils.tests.test_config_utils.TestClass",
        )
        self.assertEqual(
            get_class_config_method(TestClass),
            "torchrecipes.utils.tests.test_config_utils.TestClass.from_config",
        )

    def test_annotation_failures(self) -> None:
        class NotStatic:
            @config_entry
            def from_config(self, config: object) -> "NotStatic":
                return NotStatic()

        class MultipleEntries:
            @config_entry
            @staticmethod
            def from_config(config: object) -> "MultipleEntries":
                return MultipleEntries()

            @config_entry
            @staticmethod
            def from_hydra(config: object) -> "MultipleEntries":
                return MultipleEntries()

        class NoEntry:
            @staticmethod
            def from_config(config: object) -> "NoEntry":
                return NoEntry()

        self.assertIn("NotStatic", get_class_name_str(NotStatic))
        with self.assertRaises(ValueError):
            get_class_config_method(NotStatic)

        self.assertIn("MultipleEntries", get_class_name_str(MultipleEntries))
        with self.assertRaises(ValueError):
            get_class_config_method(MultipleEntries)

        self.assertIn("NoEntry", get_class_name_str(NoEntry))
        with self.assertRaises(ValueError):
            get_class_config_method(NoEntry)

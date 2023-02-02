# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from enum import auto, Enum, unique
from typing import List


class AutoName(Enum):
    @staticmethod
    def _generate_next_value_(
        name: str, start: int, count: int, last_values: List[auto]
    ) -> str:
        return name


@unique
class JobStatus(AutoName):
    """
    Training run job state.
    """

    # pyre-fixme[20]: Argument `value` expected.
    RUNNING = auto()
    # pyre-fixme[20]: Argument `value` expected.
    COMPLETED = auto()
    # pyre-fixme[20]: Argument `value` expected.
    FAILED = auto()

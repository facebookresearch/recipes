# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


#!/usr/bin/env python3

from typing import TypeVar, Iterable, Iterator

import testslide
from torch.utils.data import IterDataPipe
from torchrecipes.rec.datamodules.samplers.undersampler import (
    ProportionUnderSampler,
    DistributionUnderSampler,
)

T = TypeVar("T")


class IDP_NoLen(IterDataPipe[T]):
    def __init__(self, input_dp: Iterable[T]) -> None:
        super().__init__()
        self.input_dp = input_dp

    def __iter__(self) -> Iterator[T]:
        for i in self.input_dp:
            yield i


class TestUnderSampler(testslide.TestCase):
    def test_proportion_undersampler(self) -> None:
        n = 20
        idp = IDP_NoLen(range(n))
        self.assertTrue(
            all(
                i % 2 == 1
                for i in ProportionUnderSampler(idp, lambda x: x % 2, {0: 0.0, 1: 0.5})
            )
        )

    def test_proportion_undersampler_errors(self) -> None:
        n = 20
        idp = IDP_NoLen(range(n))
        with self.assertRaisesRegex(
            ValueError, "All proportions must be within 0 and 1."
        ):
            ProportionUnderSampler(idp, lambda x: x % 2, {0: 1.1})

    def test_distribution_undersampler(self) -> None:
        n = 20
        idp = IDP_NoLen(range(n))
        self.assertTrue(
            all(
                i % 2 == 1
                for i in DistributionUnderSampler(
                    idp, lambda x: x % 2, {0: 0.0, 1: 1.0}
                )
            )
        )

    def test_distribution_undersampler_known_input_dist(self) -> None:
        n = 20
        idp = IDP_NoLen(range(n))
        self.assertTrue(
            all(
                i % 2 == 1
                for i in DistributionUnderSampler(
                    idp, lambda x: x % 2, {0: 0.0, 1: 1.0}, {0: 0.5, 1: 0.5}
                )
            )
        )

    def test_distribution_undersampler_errors(self) -> None:
        n = 20
        idp = IDP_NoLen(range(n))
        with self.assertRaisesRegex(
            ValueError, "Only non-negative values are allowed in output_dist."
        ):
            DistributionUnderSampler(idp, lambda x: x % 2, {0: 0.0, 1: -1.0})
        with self.assertRaisesRegex(
            ValueError, "Only positive values are allowed in input_dist."
        ):
            DistributionUnderSampler(
                idp, lambda x: x % 2, {0: 0.0, 1: 1.0}, {0: 0.5, 1: 0.5, 2: 0.0}
            )
        with self.assertRaisesRegex(
            ValueError, "All keys in output_dist must be present in input_dist."
        ):
            DistributionUnderSampler(
                idp, lambda x: x % 2, {0: 0.0, 1: 0.9, 2: 0.1}, {0: 0.5, 1: 0.5}
            )

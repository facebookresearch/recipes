# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Iterable, Sequence, Any, Callable, Dict, Mapping

import hydra
from torchvision.transforms import Compose  # @manual


def build_transforms(transforms_config: Iterable[Mapping[str, Any]]) -> Compose:
    transform_list = [build_single_transform(config) for config in transforms_config]
    transform = Compose(transform_list)
    return transform


def build_single_transform(config: Mapping[str, Any]) -> Callable[..., object]:
    config = dict(config)
    if "transform" in config:
        assert isinstance(config["transform"], Sequence)
        transform_list = [
            build_single_transform(transform) for transform in config["transform"]
        ]
        transform = Compose(transform_list)
        config.pop("transform")
        return hydra.utils.instantiate(config, transform=transform)
    return hydra.utils.instantiate(config)


def build_transforms_from_dataset_config(
    dataset_conf: Dict[str, Any]
) -> Dict[str, Any]:
    """
    This function converts transform config to transform callable,
    then update the dataset config to use the generated callables.
    """
    transform_conf = dataset_conf.get("transform", None)
    target_transform_conf = dataset_conf.get("target_transform", None)
    transforms_conf = dataset_conf.get("transforms", None)

    if transform_conf is not None:
        dataset_conf["transform"] = build_transforms(transform_conf)
    if target_transform_conf is not None:
        dataset_conf["target_transform"] = build_transforms(target_transform_conf)
    if transforms_conf is not None:
        dataset_conf["transforms"] = build_transforms(transforms_conf)

    return dataset_conf

#!/usr/bin/env python3

import torchvision  # noqa


def register_components() -> None:
    """
    Imports all python files in the folder to trigger the
    code to register them to Hydra's ConfigStore.
    """

    from torchrecipes.vision.image_classification.callbacks import (
        register_components as callback_components,
    )

    callback_components()

    import torchrecipes.vision.image_classification.module.image_classification  # noqa
    import torchrecipes.vision.image_classification.train_app  # noqa

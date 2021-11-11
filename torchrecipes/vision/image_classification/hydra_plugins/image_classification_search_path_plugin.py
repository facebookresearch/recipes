# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin


class ImageClassificationSearchPathPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        # Appends the search path for this plugin to the end of the search path
        search_path.append(
            "image-classification-search-path-plugin",
            "pkg://torchrecipes.vision.image_classification.conf",
        )

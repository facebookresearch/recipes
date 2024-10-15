#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Module for text generation with a torchscript module containing both transform and model
IT DOES NOT SUPPORT BATCH!
"""

import logging
from abc import ABC

import torch
from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)


class TextGenerator(BaseHandler, ABC):
    """
    TextGenerator handler class. This handler takes a text (string) and
    as input and returns the generated text.
    """

    def handle(self, data, context):
        """
        Handle user's request, extract the text and return generated text.
        Batch processing is not supported. Only the first request in a batch will
        be handled.
        Example data:
            [
                "body": "hello world"
            ]
        """
        text = data[0].get("body")
        # Decode text if not a str but bytes or bytearray
        if isinstance(text, (bytes, bytearray)):
            text = text.decode("utf-8")
        # torchserve requires output to be a list
        return [self.model(text)]

    def _load_torchscript_model(self, model_pt_path):
        """Loads the PyTorch model and returns the NN model object.
        Args:
            model_pt_path (str): denotes the path of the model file.
        Returns:
            (NN Model Object) : Loads the model object.
        """
        model = torch.jit.load(model_pt_path, map_location=self.device)
        model.set_device(str(self.device))
        return model

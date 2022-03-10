# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import os
import tempfile
import unittest
import uuid

from torch.distributed.launcher.api import elastic_launch, LaunchConfig
from torchrec import test_utils
from torchrecipes.rec.datamodules.tests.utils import create_dataset_tsv
from torchrecipes.rec.dlrm_main import main


class MainTest(unittest.TestCase):
    @classmethod
    def _run_trainer(cls) -> None:
        num_days = 1
        num_days_test = 1
        dataset_path: str = tempfile.mkdtemp()
        with create_dataset_tsv(
            num_days=num_days, num_days_test=num_days_test, dataset_path=dataset_path
        ):
            tensorboard_save_dir: str = tempfile.mkdtemp()
            main(
                [
                    "--limit_train_batches",
                    "5",
                    "--limit_val_batches",
                    "5",
                    "--limit_test_batches",
                    "5",
                    "--over_arch_layer_sizes",
                    "8,1",
                    "--dense_arch_layer_sizes",
                    "8,8",
                    "--embedding_dim",
                    "8",
                    "--num_embeddings",
                    "64",
                    "--tensorboard_save_dir",
                    tensorboard_save_dir,
                    "--dataset_path",
                    dataset_path,
                ]
            )

    @test_utils.skip_if_asan
    def test_main_function(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            lc = LaunchConfig(
                min_nodes=1,
                max_nodes=1,
                nproc_per_node=2,
                run_id=str(uuid.uuid4()),
                rdzv_backend="c10d",
                rdzv_endpoint=os.path.join(tmpdir, "rdzv"),
                rdzv_configs={"store_type": "file"},
                start_method="spawn",
                monitor_interval=1,
                max_restarts=0,
            )

            elastic_launch(config=lc, entrypoint=self._run_trainer)()

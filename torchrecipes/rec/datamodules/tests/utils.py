# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import contextlib
import csv
import os
import random
from typing import Generator, List

INT_FEATURE_COUNT = 13
CAT_FEATURE_COUNT = 26


@contextlib.contextmanager
def create_dataset_tsv(
    num_rows: int = 10000,
    train: bool = True,
    # pyre-fixme[9]: dataset_path is declared to have type `str` but is used as type `None`.
    dataset_path: str = None,
    num_days: int = 1,
    num_days_test: int = 0,
    is_kaggle: bool = False,
) -> Generator[List[str], None, None]:
    """Util function to create the dataset tsv locally following the patern of the criteo dataset
    Args:
        num_rows: number of rows we want to create in this dataset
        train: if it's training dataset which will determine the generation of labels
        dataset_path: the path to create the dataset, required as an input
        num_days: number of days (out of 25) of data to use for train/validation
            only valid for criteo 1tb, as kaggle only have 1 train file
        num_days_test: number of days (out of 25) of data to use for testing
            only valid for criteo 1tb, the test data of kaggle does not label, thus not useable
        is_kaggle: if we generate Kaggle data or not

    Examples:
        >>> with create_dataset_tsv(
            num_days=num_days, num_days_test=num_days_test, dataset_path=dataset_path
        ) as _:
        >>> dm = CriteoDataModule(
                num_days=1,
                batch_size=3,
                num_days_test=0,
                num_workers=0,
                dataset_path=dataset_path,
            )
    """
    if is_kaggle is False:
        filenames = [f"day_{day}.tsv" for day in range(num_days + num_days_test)]
    else:
        filenames = ["train.txt", "test.txt"]
    paths = [os.path.join(dataset_path, filename) for filename in filenames]
    for path in paths:
        with open(path, "w") as f:
            rows = []
            for _ in range(num_rows):
                row = []
                if train:
                    row.append(str(random.randint(0, 1)))
                row += [
                    *(str(random.randint(0, 100)) for _ in range(INT_FEATURE_COUNT)),
                    *(
                        ("%x" % abs(hash(str(random.randint(0, 1000))))).zfill(8)[:8]
                        for _ in range(CAT_FEATURE_COUNT)
                    ),
                ]

                rows.append(row)
            # pyre-ignore[6]
            cf = csv.writer(f, delimiter="\t")
            cf.writerows(rows)
    yield paths

from typing import List, Tuple

import torch
from torch.utils.data import Dataset


SampleType = Tuple[int, torch.Tensor, List[torch.Tensor]]


class TestDataset(Dataset):
    def __init__(self, root_dir: str = "test"):
        self.root_dir = root_dir

    def __len__(self) -> int:
        return 10

    def __getitem__(self, key: int) -> SampleType:
        """Load the n-th sample from the dataset.
        Args:
            key (int): The index of the sample to be loaded
        Returns:
            (int, Tensor, List[Tensor]): ``(sample_rate, mix_waveform, list_of_source_waveforms)``
        """
        return 8000, torch.rand(1, 24000), [torch.rand(1, 24000), torch.rand(1, 24000)]

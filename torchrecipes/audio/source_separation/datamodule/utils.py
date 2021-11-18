from functools import partial
from typing import (
    List,
    Tuple,
)

import torch
from torch import Tensor


def get_collate_fn(
    mode: str = "train",
    sample_rate: int = 16000,
    duration: int = 4,
):
    if mode == "train":
        return partial(collate_fn, sample_rate=sample_rate, duration=duration)
    return partial(collate_fn, sample_rate=sample_rate, duration=-1)


def collate_fn(
    samples: List[Tensor],
    sample_rate: int,
    duration: int
) -> Tuple[Tensor, Tensor, Tensor]:
    if duration == -1:
        target_num_frames = max(s[1].shape[-1] for s in samples)
    else:
        target_num_frames = int(duration * sample_rate)

    mixes, srcs, masks = [], [], []
    for sample in samples:
        mix, src, mask = _fix_num_frames(sample, target_num_frames, sample_rate, random_start=True)

        mixes.append(mix)
        srcs.append(src)
        masks.append(mask)

    return (torch.stack(mixes, 0), torch.stack(srcs, 0), torch.stack(masks, 0))


def _fix_num_frames(sample: torch.Tensor, target_num_frames: int, sample_rate: int, random_start=False):
    """Ensure waveform has exact number of frames by slicing or padding"""
    mix = sample[1]  # [1, time]
    src = torch.cat(sample[2], 0)  # [num_sources, time]

    num_channels, num_frames = src.shape
    num_seconds = torch.div(num_frames, sample_rate, rounding_mode='floor')
    target_seconds = torch.div(target_num_frames, sample_rate, rounding_mode='floor')
    if num_frames >= target_num_frames:
        if random_start and num_frames > target_num_frames:
            start_frame = torch.randint(num_seconds - target_seconds + 1, [1]) * sample_rate
            mix = mix[:, start_frame:]
            src = src[:, start_frame:]
        mix = mix[:, :target_num_frames]
        src = src[:, :target_num_frames]
        mask = torch.ones_like(mix)
    else:
        num_padding = target_num_frames - num_frames
        pad = torch.zeros([1, num_padding], dtype=mix.dtype, device=mix.device)
        mix = torch.cat([mix, pad], 1)
        src = torch.cat([src, pad.expand(num_channels, -1)], 1)
        mask = torch.ones_like(mix)
        mask[..., num_frames:] = 0
    return mix, src, mask

import os
from typing import Optional, Any

import fsspec
from pytorch_lightning.callbacks import ModelCheckpoint


def get_filesystem(path: str, **kwargs: Any) -> fsspec.AbstractFileSystem:
    """Returns the appropriate filesystem to use when handling the given path."""
    if "://" in path:
        # use the fileystem from the protocol specified
        return fsspec.filesystem(path.split(":", 1)[0], **kwargs)
    # use local filesystem
    return fsspec.filesystem("file", **kwargs)


def find_last_checkpoint_path(checkpoint_dir: Optional[str]) -> Optional[str]:
    """Takes in a checkpoint directory path, looks for a last.ckpt checkpoint inside,
    and returns the full path that we can use for resuming from that checkpoint.

    Args:
        checkpoint_dir: Path where the model file(s) are saved.

    Returns:
        Full path for the last model checkpoint from the given checkpoint directory.
    """
    if checkpoint_dir is None:
        return None
    checkpoint_file_name = (
        f"{ModelCheckpoint.CHECKPOINT_NAME_LAST}{ModelCheckpoint.FILE_EXTENSION}"
    )
    last_checkpoint_filepath = os.path.join(checkpoint_dir, checkpoint_file_name)
    if not get_filesystem(last_checkpoint_filepath).exists(last_checkpoint_filepath):
        return None

    return last_checkpoint_filepath

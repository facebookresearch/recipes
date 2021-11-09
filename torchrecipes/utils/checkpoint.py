import os
from typing import Optional

from fsspec.core import url_to_fs
from pytorch_lightning.callbacks import ModelCheckpoint


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
    fs, _ = url_to_fs(last_checkpoint_filepath)
    if not fs.exists(last_checkpoint_filepath):
        return None

    return last_checkpoint_filepath

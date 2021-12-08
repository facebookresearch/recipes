import torch
from torchrecipes.audio.source_separation.loss import utils


def si_sdr_loss(
    estimate: torch.Tensor, reference: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Compute the Si-SDR loss.
    Args:
        estimate (torch.Tensor): Estimated source signals.
            Tensor of dimension (batch, speakers, time)
        reference (torch.Tensor): Reference (original) source signals.
            Tensor of dimension (batch, speakers, time)
        mask (torch.Tensor): Mask to indicate padded value (0) or valid value (1).
            Tensor of dimension (batch, 1, time)
    Returns:
        torch.Tensor: Si-SDR loss. Tensor of dimension (batch, )
    """
    estimate = estimate - estimate.mean(axis=2, keepdim=True)
    reference = reference - reference.mean(axis=2, keepdim=True)

    si_sdri = utils.sdr_pit(estimate, reference, mask=mask)
    return -si_sdri.mean()

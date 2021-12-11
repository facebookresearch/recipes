import torch

from ..loss import utils


def sisdri_metric(
    estimate: torch.Tensor,
    reference: torch.Tensor,
    mix: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Compute the improvement of scale-invariant SDR. (SI-SDRi).
    Args:
        estimate (torch.Tensor): Estimated source signals.
            Tensor of dimension (batch, speakers, time)
        reference (torch.Tensor): Reference (original) source signals.
            Tensor of dimension (batch, speakers, time)
        mix (torch.Tensor): Mixed souce signals, from which the setimated signals were generated.
            Tensor of dimension (batch, speakers == 1, time)
        mask (torch.Tensor): Mask to indicate padded value (0) or valid value (1).
            Tensor of dimension (batch, 1, time)
    Returns:
        torch.Tensor: Improved SI-SDR. Tensor of dimension (batch, )
    References:
        - Conv-TasNet: Surpassing Ideal Time--Frequency Magnitude Masking for Speech Separation
        Luo, Yi and Mesgarani, Nima
        https://arxiv.org/abs/1809.07454
    """
    with torch.no_grad():
        estimate = estimate - estimate.mean(axis=2, keepdim=True)
        reference = reference - reference.mean(axis=2, keepdim=True)
        mix = mix - mix.mean(axis=2, keepdim=True)

        si_sdri = utils.sdri(estimate, reference, mix, mask=mask)

    return si_sdri.mean().item()


def sdri_metric(
    estimate: torch.Tensor,
    reference: torch.Tensor,
    mix: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Compute the improvement of SDR. (SDRi).
    Args:
        estimate (torch.Tensor): Estimated source signals.
            Tensor of dimension (batch, speakers, time)
        reference (torch.Tensor): Reference (original) source signals.
            Tensor of dimension (batch, speakers, time)
        mix (torch.Tensor): Mixed souce signals, from which the setimated signals were generated.
            Tensor of dimension (batch, speakers == 1, time)
        mask (torch.Tensor): Mask to indicate padded value (0) or valid value (1).
            Tensor of dimension (batch, 1, time)
    Returns:
        torch.Tensor: Improved SDR. Tensor of dimension (batch, )
    References:
        - Conv-TasNet: Surpassing Ideal Time--Frequency Magnitude Masking for Speech Separation
        Luo, Yi and Mesgarani, Nima
        https://arxiv.org/abs/1809.07454
    """
    with torch.no_grad():
        sdri = utils.sdri(estimate, reference, mix, mask=mask)
    return sdri.mean().item()

import torch
from torch.distributions import Categorical, kl_divergence
from torch.nn import functional as F


def _l2_normalize(input: torch.Tensor) -> torch.Tensor:
    return input / (input.norm(p=2, dim=1, keepdim=True) + 1e-8)


def _kl(input: torch.Tensor,
        target: torch.Tensor) -> torch.Tensor:
    return kl_divergence(Categorical(logits=input), Categorical(logits=target)).mean()


def mse_with_logits(input: torch.Tensor,
                    target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(input.softmax(dim=1), target)

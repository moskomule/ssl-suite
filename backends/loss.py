import torch
from torch.distributions import Categorical, kl_divergence
from torch.nn import functional as F


def normalize(input: torch.Tensor) -> torch.Tensor:
    # since PyTorch does not support multi-dimensional max
    size = input.size()
    input = input.flatten(1)
    input.div_(1e-12 + input.abs().max(dim=1, keepdim=True)[0])
    input.div_((1e-6 + input).norm(p=2, dim=1, keepdim=True))
    return input.view(size)


def kl_div(input: torch.Tensor,
           target: torch.Tensor) -> torch.Tensor:
    return kl_divergence(Categorical(logits=input), Categorical(logits=target)).mean()


def mse_with_logits(input: torch.Tensor,
                    target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(input.softmax(dim=1), target)

import torch
from homura.modules.functional.loss import cross_entropy_with_softlabels
from torch.distributions import Beta

from backends.loss import mse_with_logits
from backends.utils import get_task, SSLTrainerBase


class ICTTrainer(SSLTrainerBase):

    def labeled(self,
                input: torch.Tensor,
                target: torch.Tensor):
        target = self.to_onehot(target, self.num_classes)
        input, target = self.mixup(input, target)
        output = self.model(input)
        loss = self.loss_f[0](output, target)
        return output, loss

    def unlabeled(self,
                  input: torch.Tensor):
        with torch.no_grad():
            expected = self.ema(input).softmax(dim=-1)
            input, expected = self.mixup(input, expected)
        output = self.model(input)
        loss = self.loss_f[1](output, expected)
        return output, loss

    def mixup(self,
              input: torch.Tensor,
              target: torch.Tensor):
        if not torch.is_tensor(self.beta):
            self.beta = torch.tensor(self.beta).to(self.device)
        gamma = Beta(self.beta, self.beta).sample((input.size(0), 1, 1, 1))
        perm = torch.randperm(input.size(0))
        perm_input = input[perm]
        perm_target = target[perm]
        input.mul_(gamma).add_(perm_input.mul_(1 - gamma))
        gamma = gamma.view(-1, 1)
        target.mul_(gamma).add_(perm_target.mul_(1 - gamma))
        return input, target


if __name__ == '__main__':
    import hydra

    hydra.main('config/ict.yaml')(
        get_task(ICTTrainer, [cross_entropy_with_softlabels, mse_with_logits])
    )()

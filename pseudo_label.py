import torch
from homura.modules import cross_entropy_with_softlabels
from torch.nn import functional as F

from backends.utils import SSLTrainerBase, disable_bn_stats, get_task


class PseudoLabelTrainer(SSLTrainerBase):

    def labeled(self,
                input: torch.Tensor,
                target: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        target = self.to_onehot(target, self.smoothing)
        output = self.model(input)
        loss = self.loss_f(output, target)
        return output, loss

    def unlabeled(self,
                  input: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        with disable_bn_stats(self.model):
            u_output = self.model(input)
        u_loss = F.cross_entropy(u_output, u_output.argmax(dim=1), reduction='none')
        u_loss = ((u_output.softmax(dim=1) > self.threshold).any(dim=1).float() * u_loss).mean()
        return u_output, u_loss


if __name__ == "__main__":
    import hydra

    hydra.main('config/pseudo_label.yaml')(
        get_task(PseudoLabelTrainer, cross_entropy_with_softlabels)
    )()

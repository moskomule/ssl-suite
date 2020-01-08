import torch
from homura.modules import cross_entropy_with_softlabels
from torch.distributions import Categorical
from torch.nn import functional as F

from backends.loss import kl_div, normalize
from backends.utils import SSLTrainerBase, disable_bn_stats, get_task


class VATTrainer(SSLTrainerBase):
    def labeled(self,
                input: torch.Tensor,
                target: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        output = self.model(input)
        target = self.to_onehot(target, self.smoothing)
        s_loss = self.loss_f[0](output, target)
        return output, s_loss

    def unlabeled(self,
                  input: torch.Tensor) -> (None, torch.Tensor, torch.Tensor):
        with disable_bn_stats(self.model):
            logits = self.model(input)
            u_loss = self.vat_loss(input, logits.clone().detach())
            e_loss = Categorical(logits=logits).entropy().mean()
        return None, u_loss, e_loss

    def vat_loss(self,
                 input: torch.Tensor,
                 logits: torch.Tensor) -> torch.Tensor:
        d = normalize(input.clone().normal_())
        d.requires_grad_(True)
        pred_hat = self.model(input + self.xi * d)
        adv_loss = kl_div(logits, pred_hat)
        d_grad, = torch.autograd.grad([adv_loss], [d])
        d = normalize(d_grad)
        self.model.zero_grad()
        pred_hat = self.model(input + self.eps * d)
        return kl_div(logits, pred_hat)


if __name__ == "__main__":
    import hydra

    hydra.main('config/vat.yaml')(
        get_task(VATTrainer, [cross_entropy_with_softlabels, F.cross_entropy])
    )()

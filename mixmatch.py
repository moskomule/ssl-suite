from typing import Tuple

import torch
from homura import Map
from homura.modules.functional import cross_entropy_with_softlabels
from homura.vision import mixup
from numpy.random import beta
from torch.nn import functional as F

from backends.utils import disable_bn_stats, get_task, SSLTrainerBase


class MixmatchTrainer(SSLTrainerBase):

    def iteration(self, data):
        if self.is_train:
            with torch.no_grad():
                with disable_bn_stats(self.model):
                    l_x, l_y, u_x, u_y = self.data_handle(data)
                labeled_size = l_x.size(0)

                x = torch.cat([l_x, u_x], dim=0)
                y = torch.cat([l_y, u_y], dim=0)
                gamma = beta(self.gamma, self.gamma)
                x, y = mixup(x, y, max(gamma, 1 - gamma))

            o = self.model(x)
            loss = self.loss_f[0](o[:labeled_size], y[:labeled_size]) + \
                   self.coef * self.loss_f[1](o[labeled_size:].softmax(dim=1), y[labeled_size:])

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.ema.update()
            return Map(loss=loss, output=o[:labeled_size])
        else:
            return super().iteration(data)

    def data_handle(self,
                    data: Tuple) -> Tuple:
        input, target, u_x1, u_x2, _ = data
        u_x, u_y = self.sharpen((u_x1, u_x2))
        return input, self.to_onehot(target), u_x, u_y

    def sharpen(self,
                input: torch.Tensor or Tuple) -> Tuple[torch.Tensor, torch.Tensor]:
        u_b = torch.cat(input, dim=0)
        q_b = (self.model(input[0]).softmax(dim=1) + self.model(input[1]).softmax(dim=1)) / 2
        q_b.pow_(1 / self.temperature).div_(q_b.sum(dim=1, keepdim=True))
        return u_b, q_b.repeat(2, 1)


if __name__ == "__main__":
    import hydra

    hydra.main('config/mixmatch.yaml')(
        get_task(MixmatchTrainer,
                 [cross_entropy_with_softlabels, F.mse_loss])
    )()

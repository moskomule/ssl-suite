import torch
from homura.modules import cross_entropy_with_softlabels, to_onehot
from torch.nn import functional as F

from backends.utils import SSLTrainerBase, disable_bn_stats, get_task


class MeanTeacherTrainer(SSLTrainerBase):

    def labeled(self,
                input: torch.Tensor,
                target: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        output = self.model(input)
        target = to_onehot(target, self.num_classes)
        target -= self.smoothing * (target - 1 / self.num_classes)
        loss = self.loss_f(output, target)
        return output, loss

    def unlabeled(self,
                  input1: torch.Tensor,
                  input2: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        with disable_bn_stats(self.model):
            o1 = self.model(input1)
        o2 = self.ema(input2)
        return o1, F.mse_loss(o1.softmax(dim=1), o2.softmax(dim=1))


if __name__ == "__main__":
    import hydra

    hydra.main('config/mean_teacher.yaml')(
        get_task(MeanTeacherTrainer, cross_entropy_with_softlabels)
    )()

from functools import partial
from statistics import median
from typing import Tuple, Mapping

import hydra
import torch
import torch.nn.functional as F
from homura import optim, callbacks, reporters, trainers, Map

from backends.supervised_backends import get_dataloaders
from backends.utils import EMAModel
from backends.wrn import wrn28_2


class SupervisedTrainer(trainers.TrainerBase):
    def __init__(self, *args, **kwargs):
        super(SupervisedTrainer, self).__init__(*args, **kwargs)
        self.ema = kwargs['ema_model'](self.model)

    def iteration(self,
                  data: Tuple[torch.Tensor]) -> Mapping[str, torch.Tensor]:
        input, target = data
        if self.is_train:
            output = self.model(input)
        else:
            output = self.ema(input)
        loss = self.loss_f(output, target)
        if self.is_train:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.ema.update()
        return Map(loss=loss, output=output)


@hydra.main("config/supervised.yaml")
def main(cfg):
    model = wrn28_2(num_classes=10)
    train_loader, test_loader = get_dataloaders(cfg.data.name,
                                                cfg.data.batch_size,
                                                cfg.data.train_size,
                                                cfg.data.random_state)
    optimizer = optim.Adam(lr=cfg.optim.lr)
    tq = reporters.TQDMReporter(range(cfg.optim.epochs))
    c = [callbacks.AccuracyCallback(),
         callbacks.LossCallback(),
         reporters.IOReporter("."),
         tq]

    with SupervisedTrainer(model,
                           optimizer,
                           F.cross_entropy,
                           callbacks=c,
                           ema_model=partial(EMAModel, ema_rate=cfg.model.ema_rate,
                                             weight_decay=cfg.optim.wd * cfg.optim.lr),
                           ) as trainer:
        for _ in tq:
            trainer.train(train_loader)
            trainer.test(test_loader)
        trainer.logger.info(f"test accuracy: {median(c[0].history['test'][-20:])}")


if __name__ == '__main__':
    main()

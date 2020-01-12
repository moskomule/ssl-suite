import contextlib
from copy import deepcopy
from functools import partial
from itertools import cycle
from statistics import median
from typing import Tuple, Mapping, Callable, Optional

import torch
from homura import optim, trainers, reporters, callbacks, Map, get_args, lr_scheduler
from homura.liblog import get_logger
from homura.modules import exponential_moving_average_, to_onehot
from torch import nn
from torch.nn import functional as F

from backends.data import get_dataloader
from backends.wrn import wrn28_2

logger = get_logger(__file__)


class PackedLoader(object):
    def __init__(self, trusted_loader, untrusted_loader):
        self.l_loaders = trusted_loader
        self.u_loaders = untrusted_loader
        self._size = len(untrusted_loader)

    def __len__(self):
        return self._size

    def __iter__(self):
        for l, u in zip(cycle(self.l_loaders), self.u_loaders):
            yield list(l) + list(u)


@contextlib.contextmanager
def disable_bn_stats(model: nn.Module):
    def f(m: nn.Module):
        if hasattr(m, 'track_running_stats'):
            m.track_running_stats ^= True

    model.apply(f)
    yield
    model.apply(f)


class EMAModel(object):
    def __init__(self,
                 model: nn.Module,
                 ema_rate: float,
                 weight_decay: float):
        self.model = model
        self.ema_model = deepcopy(self.model)
        self.ema_model.requires_grad_(False)

        def f(m: nn.Module):
            if hasattr(m, 'track_running_stats'):
                m.track_running_stats = False

        self.ema_model.apply(f)
        self.ema_model.eval()
        self.ema_rate = ema_rate
        self.weight_decay = weight_decay

    def __call__(self,
                 input: torch.Tensor) -> torch.Tensor:
        return self.ema_model(input)

    def update(self):
        for e_b, o_b in zip(self.ema_model.buffers(), self.model.buffers()):
            # e_b.data.copy_(o_b.data)
            if e_b.data.dtype == torch.float32:
                exponential_moving_average_(e_b.data, o_b.data, self.ema_rate)
                # self._apply_weight_decay(o_b.data)

        for (e_n, e_p), (o_n, o_p) in zip(self.ema_model.named_parameters(), self.model.named_parameters()):
            exponential_moving_average_(e_p.data, o_p.data, self.ema_rate)
            if 'bn' not in e_n:
                self._apply_weight_decay(o_p.data)

    def _apply_weight_decay(self,
                            param: torch.Tensor):
        if self.weight_decay > 0:
            param.mul_(1 - self.weight_decay)


def get_components(cfg):
    labeled_loader, unlabeled_loader, val_loader, test_loader = get_dataloader(cfg.data.name,
                                                                               cfg.data.labeled_size,
                                                                               cfg.data.unlabeled_size,
                                                                               cfg.data.val_size,
                                                                               cfg.data.batch_size,
                                                                               cfg.data.random_state,
                                                                               download=cfg.data.download,
                                                                               pilaugment=cfg.data.get('pilaugment',
                                                                                                       False)
                                                                               )

    model = wrn28_2(num_classes=6 if cfg.data.name == "animal" else 10)
    optimizer = {'adam': optim.Adam(lr=cfg.optim.lr),
                 'sgd': optim.SGD(lr=cfg.optim.lr, momentum=0.9)}[cfg.optim.name]
    scheduler = {'adam': None,
                 'sgd': lr_scheduler.CosineAnnealingWithWarmup(cfg.optim.epochs,
                                                               4, cfg.optim.epochs // 100)}[cfg.optim.name]
    ema_model = partial(EMAModel, ema_rate=cfg.model.ema_rate, weight_decay=cfg.optim.wd * cfg.optim.lr)
    num_classes = {"animal": 6, "cifar100": 100, "tinyimagenet": 200}.get(cfg.data.name, 10)
    tq = reporters.TQDMReporter(range(cfg.optim.epochs))
    _callbacks = [callbacks.AccuracyCallback(),
                  callbacks.LossCallback(),
                  reporters.IOReporter("."),
                  reporters.TensorboardReporter("."), tq]
    return PackedLoader(labeled_loader, unlabeled_loader), val_loader, test_loader, model, optimizer, \
           scheduler, ema_model, num_classes, tq, _callbacks


def get_task(trainer,
             loss_f: Tuple[Callable] or Callable):
    import warnings

    warnings.simplefilter("ignore")

    def main(cfg):
        print(get_args())
        print(cfg.pretty())
        (train_loader, val_loader, test_loader, model, optimizer, scheduler, ema_model, num_classes,
         tq, callbacks) = get_components(cfg)

        with trainer(model,
                     optimizer,
                     loss_f,
                     scheduler=scheduler,
                     callbacks=callbacks,
                     ema_model=ema_model,
                     num_classes=num_classes,
                     cfg=cfg.model
                     ) as t:
            for ep in tq:
                t.train(train_loader)
                t.test(val_loader, mode='val')
                t.test(test_loader)
            t.logger.info(f"test accuracy: {median(callbacks[0].history['test'][-20:])}")

        return 1 - callbacks[0].history["val"][-1]

    return main


class SSLTrainerBase(trainers.TrainerBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ema = kwargs['ema_model'](self.model)
        for k, v in kwargs['cfg'].items():
            setattr(self, k, v)

    def iteration(self, data: Tuple[torch.Tensor]) -> Mapping:
        if self.is_train:
            if len(data) == 4:
                # do not augment by PIL
                l_input, l_target, u_input, _ = data
                u_out = self.unlabeled(u_input)
            elif len(data) == 5:
                # augment by PIL
                l_input, l_target, u_input1, u_input2, _ = data
                u_out = self.unlabeled(u_input1, u_input2)
            else:
                raise NotImplementedError

            output, l_loss = self.labeled(l_input, l_target)
            if len(u_out) == 2:
                # not entropy regularization
                u_output, u_loss = u_out
                loss = l_loss + self.coef * u_loss
            elif len(u_out) == 3:
                # entropy regularization
                u_output, u_loss, e_loss = u_out
                loss = l_loss + self.coef * u_loss + self.coef_ent * e_loss
            else:
                raise NotImplementedError

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.ema.update()
        else:
            input, target = data
            output = self.ema(input)
            loss = F.cross_entropy(output, target)
        return Map(loss=loss, output=output)

    def labeled(self,
                input: torch.Tensor,
                target: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        # handle labeled data
        raise NotImplementedError

    def unlabeled(self,
                  *input: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        # handle unlabeled data
        raise NotImplementedError

    @property
    def coef(self) -> float:
        # coeficient at every epoch
        return self.coef_max * min(1, self.epoch / self.coef_iters)

    def to_onehot(self,
                  target: torch.Tensor,
                  smoothing: Optional[float] = None):
        target = to_onehot(target, self.num_classes)
        if smoothing is not None:
            target -= self.smoothing * (target - 1 / self.num_classes)
        return target

from statistics import median

import torch.nn.functional as F
from homura import optim, lr_scheduler, callbacks, reporters, trainers
from homura.vision.data.loaders import cifar10_loaders

from backends.wrn import wrn28_2


def main():
    model = wrn28_2(num_classes=10)
    train_loader, test_loader = cifar10_loaders(128)
    optimizer = optim.SGD(lr=1e-1, momentum=0.9, weight_decay=5e-4)
    scheduler = lr_scheduler.MultiStepLR([100, 150], gamma=0.2)
    tq = reporters.TQDMReporter(range(200), verb=True)
    c = [callbacks.AccuracyCallback(),
         callbacks.LossCallback(),
         reporters.IOReporter("."),
         tq]

    with trainers.SupervisedTrainer(model, optimizer, F.cross_entropy, callbacks=c,
                                    scheduler=scheduler) as trainer:
        for _ in tq:
            trainer.train(train_loader)
            trainer.test(test_loader)
        trainer.logger.info(f"test accuracy: {median(c[0].history['test'][-20:])}")


if __name__ == '__main__':
    main()

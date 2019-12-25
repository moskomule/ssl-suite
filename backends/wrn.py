# WideResNet proposed in http://arxiv.org/abs/1605.07146
# modified as https://github.com/google-research/mixmatch/blob/master/libml/models.py
#
import torch
import torch.nn as nn
import torch.nn.functional as F

bn_kwargs = dict(momentum=1 - 0.999, eps=1e-3)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=True)


class WideBasicModule(nn.Module):

    def __init__(self,
                 in_planes: int,
                 planes: int,
                 dropout_rate: float,
                 stride: int = 1,
                 preact: bool = False):
        super(WideBasicModule, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes, **bn_kwargs)
        self.conv1 = nn.Conv2d(in_planes, planes, (3, 3), 1, 1)
        self.bn2 = nn.BatchNorm2d(planes, **bn_kwargs)
        self.conv2 = nn.Conv2d(planes, planes, (3, 3), stride, 1)
        self.preact = preact
        self.shortcut = None
        if stride != 1 or in_planes != planes:
            self.shortcut = conv1x1(in_planes, planes, stride)

    def forward(self,
                input: torch.Tensor):

        x = F.leaky_relu(self.bn1(input), 0.1)
        if self.preact:
            residual = x
        else:
            residual = input
        x = self.conv2(F.leaky_relu(self.bn2(self.conv1(x)), 0.1))
        if self.shortcut is not None:
            residual = self.shortcut(residual)
        return x + residual


class WideResNet(nn.Module):
    """WideResNet for CIFAR data.
    """

    def __init__(self, num_classes, depth, widen_factor, dropout_rate, base=16):
        super(WideResNet, self).__init__()
        self.in_planes = base

        assert ((depth - 4) % 6 == 0), "depth should be 6n+4"
        n = (depth - 4) // 6
        k = widen_factor

        num_stages = [base, base * k, base * k * 2, base * k * 4]

        self.conv1 = conv3x3(3, num_stages[0])
        self.layer1 = self._wide_layer(WideBasicModule, num_stages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(WideBasicModule, num_stages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(WideBasicModule, num_stages[3], n, dropout_rate, stride=2)
        self.bn = nn.BatchNorm2d(num_stages[3], **bn_kwargs)
        self.fc = nn.Linear(num_stages[3], num_classes)
        self.initialize()

    def initialize(self):
        # following author's implementation
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(getattr(m, "running_mean"), 0)
                nn.init.constant_(getattr(m, "running_var"), 1)
            elif isinstance(m, nn.Conv2d):
                _out, _, _k, _ = m.weight.size()
                nn.init.normal_(m.weight, 0, 1 / (_k * _k * _out))
                if hasattr(m, 'bias'):
                    nn.init.constant_(m.bias, 0)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1] * int(num_blocks - 1)
        layers = []

        for i, stride in enumerate(strides):
            layers.append(block(self.in_planes, planes, dropout_rate, stride, i == 0))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = F.leaky_relu(self.bn(x), 0.1)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def wrn28_10(num_classes=10, dropout_rate=0) -> WideResNet:
    model = WideResNet(depth=28, widen_factor=10, dropout_rate=dropout_rate, num_classes=num_classes)
    return model


def wrn28_2(num_classes=10, dropout_rate=0) -> WideResNet:
    model = WideResNet(depth=28, widen_factor=2, dropout_rate=dropout_rate, num_classes=num_classes)
    return model

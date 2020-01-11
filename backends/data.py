import shutil
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from homura.liblog import get_logger
from homura.utils.reproducibility import set_seed
from torch.utils.data import RandomSampler, DataLoader, ConcatDataset
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, SVHN, VisionDataset, ImageFolder, utils

logger = get_logger(__file__)


class TinyImageNet(ImageFolder):
    # tiny imagenet dataset

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        root = Path(root).expanduser()
        if not root.exists() and download:
            root.mkdir(exist_ok=True, parents=True)
            self.download(root)
        if root.exists() and download:
            print("Files already downloaded")
        root = root / "train" if train else root / "val"
        super(TinyImageNet, self).__init__(root, transform=transform, target_transform=target_transform)
        self.data = [self.loader(img) for img, _ in self.samples]

    def download(self,
                 root: Path):
        utils.download_and_extract_archive(url="http://cs231n.stanford.edu/tiny-imagenet-200.zip",
                                           download_root=root)
        temp_dir = "tiny-imagenet-200"
        shutil.move(str(root / temp_dir / "train"), str(root))
        shutil.move(str(root / temp_dir / "val"), str(root))
        val_dir = root / "val"
        with (val_dir / "val_annotations.txt").open() as f:
            val_anns = f.read()
        name_to_cls = {i[0]: i[1] for i in
                       [l.split() for l in val_anns.strip().split("\n")]}
        for name, cls in name_to_cls.items():
            (val_dir / cls).mkdir(exist_ok=True)
            shutil.move(str(val_dir / "images" / name), str(val_dir / cls))
        shutil.rmtree(str(root / temp_dir))

    def __len__(self):
        return len(self.data)


def getitem(self, index):
    img, target = self.data[index], self.targets[index]

    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    if hasattr(self, 'pair_img'):
        return self.transform(img), self.transform(img), target
    else:
        return self.transform(img), target


class OriginalSVHN(SVHN):
    def __init__(self,
                 root,
                 train=True,
                 transform=None,
                 target_transform=None,
                 download=False):
        super(OriginalSVHN, self).__init__(root, split="train" if train else "test", transform=transform,
                                           target_transform=target_transform, download=download)
        self.data = [Image.fromarray(np.transpose(img, (1, 2, 0))) for img in self.data]
        self.targets = self.labels


def get_dataloader(dataset: str,
                   labeled_size: int,
                   unlabeled_size: int,
                   val_size: int,
                   batch_size: int,
                   random_state: int = -1,
                   download: bool = False,
                   balanced: bool = True,
                   pilaugment: bool = False) -> (DataLoader, DataLoader, DataLoader, DataLoader):
    with set_seed(random_state):
        labeled_set, unlabeled_set, val_set, test_set = _get_dataset(dataset, labeled_size, unlabeled_size,
                                                                     val_size, download, balanced,
                                                                     pilaugment=pilaugment)
    labeled_loader = DataLoader(labeled_set, batch_size=batch_size,
                                sampler=RandomSampler(labeled_set, True, labeled_size),
                                num_workers=2, pin_memory=True)
    unlabeled_loader = DataLoader(unlabeled_set, batch_size=batch_size,
                                  sampler=RandomSampler(unlabeled_set, True, 64 * 1024),
                                  num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=2 * batch_size, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=2 * batch_size)

    log_text = f"Generate dataloaders: {dataset} of \n" \
               f">>>labeled size   {labeled_size}\n" \
               f">>>unlabeled size {unlabeled_size}\n" \
               f">>>val size       {val_size}\n"
    logger.info(log_text)
    return labeled_loader, unlabeled_loader, val_loader, test_loader


DATASETS = {"cifar10": (CIFAR10, "~/.torch/data/cifar10",
                        [transforms.ToTensor(),
                         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))],
                        [transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                         transforms.RandomHorizontalFlip()], 10),
            "cifar100": (CIFAR100, "~/.torch/data/cifar100",
                         [transforms.ToTensor(),
                          transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))],
                         [transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                          transforms.RandomHorizontalFlip()], 100),
            "svhn": (OriginalSVHN, "~/.torch/data/svhn",
                     [transforms.ToTensor(),
                      transforms.Normalize((0.4390, 0.4443, 0.4692), (0.1189, 0.1222, 0.1049))],
                     [transforms.RandomCrop(32, padding=4, padding_mode='reflect')], 10),
            "tinyimagenet": (TinyImageNet, "~/.torch/data/tinyimagenet",
                             [transforms.ToTensor(),
                              transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))],
                             [transforms.Resize(40),
                              transforms.RandomCrop(32),
                              transforms.RandomHorizontalFlip()], 200)
            }


def _get_dataset(dataset: str,
                 labeled_size: int,
                 unlabeled_size: int,
                 val_size: int,
                 download: bool,
                 balanced: bool,
                 pilaugment: bool = False) -> (VisionDataset, VisionDataset, VisionDataset, VisionDataset):
    if dataset in DATASETS.keys():
        dset, root, norm_transform, data_aug, num_cls = DATASETS.get(dataset, DATASETS["cifar10"])
        labeled_set = dset(root, train=True, transform=transforms.Compose(data_aug + norm_transform), download=download)
        test_set = dset(root, train=False, transform=transforms.Compose(norm_transform), download=download)
        labeled_set, unlabeled_set, val_set = _split_dataset(labeled_set, labeled_size, unlabeled_size, val_size,
                                                             num_cls, balanced=balanced)
        val_set.transform = transforms.Compose(norm_transform)

    else:
        raise NotImplementedError

    if pilaugment:
        if isinstance(unlabeled_set, ConcatDataset):
            for d in unlabeled_set.datasets:
                type(d).__getitem__ = getitem
                d.pair_img = True
        else:
            type(unlabeled_set).__getitem__ = getitem
            unlabeled_set.pair_img = True
        logger.info('Enable PIL augment')
    return labeled_set, unlabeled_set, val_set, test_set


def _split_dataset(dataset: VisionDataset,
                   labeled_size: int,
                   unlabeled_size: int,
                   val_size: int,
                   num_classes: int,
                   balanced: bool) -> (VisionDataset, VisionDataset, VisionDataset):
    # split given dataset into labeled, unlabeled and val

    assert labeled_size + unlabeled_size + val_size == len(dataset)
    indices = torch.randperm(len(dataset))
    dataset.data = [dataset.data[i] for i in indices]
    dataset.targets = [dataset.targets[i] for i in indices]

    labeled_set = dataset
    unlabeled_set = deepcopy(dataset)
    val_set = deepcopy(dataset)
    labeled_set.data = [labeled_set.data[i] for i in range(labeled_size)]
    unlabeled_set.data = [unlabeled_set.data[i] for i in range(labeled_size, labeled_size + unlabeled_size)]
    val_set.data = [val_set.data[i] for i in range(labeled_size + unlabeled_size,
                                                   labeled_size + unlabeled_size + val_size)]
    labeled_set.targets = [labeled_set.targets[i] for i in range(labeled_size)]
    unlabeled_set.targets = [-1 for _ in range(unlabeled_size)]
    val_set.targets = [val_set.targets[i] for i in range(labeled_size + unlabeled_size,
                                                         labeled_size + unlabeled_size + val_size)]
    return labeled_set, unlabeled_set, val_set


if __name__ == '__main__':
    # download all datasets

    _get_dataset('cifar10', 4_000, 41_000, 5_000, download=True, balanced=False)

import torch
from homura.utils.reproducibility import set_seed
from torch.utils.data import DataLoader, RandomSampler
from torchvision import transforms

from .data import DATASETS, getitem


def get_dataloaders(dataset: str,
                    batch_size: int,
                    train_size: int,
                    seed: int) -> (DataLoader, DataLoader):
    if dataset in DATASETS.keys():
        dset, root, norm_transform, data_aug, num_cls = DATASETS[dataset]
        dset.__getitem__ = getitem
        labeled_set = dset(root, train=True, transform=transforms.Compose(data_aug + norm_transform))
        with set_seed(seed):
            indices = torch.randperm(len(labeled_set))
            labeled_set.data = [labeled_set.data[i] for i in indices][:train_size]
            labeled_set.targets = [labeled_set.targets[i] for i in indices][:train_size]
        test_set = dset(root, train=False, transform=transforms.Compose(norm_transform))
        train_loader = DataLoader(labeled_set, batch_size=batch_size,
                                  sampler=RandomSampler(labeled_set, True, num_samples=batch_size * 1024),
                                  num_workers=2, pin_memory=True)
        test_loader = DataLoader(test_set, batch_size=2 * batch_size)
    else:
        raise NotImplementedError
    return train_loader, test_loader

from torch.utils.data import DataLoader, RandomSampler
from torchvision import transforms

from .data import DATASETS, getitem


def get_dataloaders(dataset: str,
                    batch_size: int) -> (DataLoader, DataLoader):
    if dataset in DATASETS.keys():
        dset, root, norm_transform, data_aug, num_cls = DATASETS[dataset]
        dset.__getitem__ = getitem
        labeled_set = dset(root, train=True, transform=transforms.Compose(data_aug + norm_transform))
        test_set = dset(root, train=False, transform=transforms.Compose(norm_transform))
        train_loader = DataLoader(labeled_set, batch_size=batch_size,
                                  sampler=RandomSampler(labeled_set, True),
                                  num_workers=2, pin_memory=True)
        test_loader = DataLoader(test_set, batch_size=2 * batch_size)
    else:
        raise NotImplementedError
    return train_loader, test_loader

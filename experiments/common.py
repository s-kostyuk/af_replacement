import torch.utils.data
import torchvision

from typing import Tuple

from misc import get_file_name_net, get_file_name_opt, get_file_name_stat


def get_device() -> torch.device:
    if torch.cuda.is_available():
        print("Using GPU computing unit")
        torch.cuda.set_device(0)
        device = torch.device('cuda:0')
        print("Cuda computing capability: {}.{}".format(
            *torch.cuda.get_device_capability(device)
        ))
    else:
        print("Using CPU computing unit")
        device = torch.device('cpu')

    return device


def get_cifar10_dataset(
        augment: bool = False
) -> Tuple[torch.utils.data.Dataset, ...]:
    if augment:
        augments = (
            # as in Keras - each second image is flipped
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
            # assuming that the values from git.io/JuHV0 were used in
            # arXiv 1801.09403
            torchvision.transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))
        )
    else:
        augments = ()

    train_set = torchvision.datasets.CIFAR10(
        root="./data/cifar",
        train=True,
        download=True,
        transform=torchvision.transforms.Compose(
            (torchvision.transforms.ToTensor(), *augments)
        )
    )

    test_set = torchvision.datasets.CIFAR10(
        root="./data/cifar",
        train=False,
        download=True,
        transform=torchvision.transforms.Compose(
            (torchvision.transforms.ToTensor(),)
        )
    )

    return train_set, test_set

"""CIFAR100 dataset utilities for federated learning."""

# flake8: noqa

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from project.task.default.dataset import (
    ClientDataloaderConfig as DefaultClientDataloaderConfig,
)
from project.task.default.dataset import (
    FedDataloaderConfig as DefaultFedDataloaderConfig,
)
from project.types.common import (
    CID,
    ClientDataloaderGen,
    FedDataloaderGen,
    IsolatedRNG,
)

from torchvision.datasets import VisionDataset
from typing import Any
from collections.abc import Callable
from PIL import Image
import numpy as np

from project.task.cifar100_classification.dataset_preparation import (
    cifar100Transformation,
)

# Use defaults for this very simple dataset
# Requires only batch size
ClientDataloaderConfig = DefaultClientDataloaderConfig
FedDataloaderConfig = DefaultFedDataloaderConfig


class TorchVision_FL(VisionDataset):
    """This is just a trimmed down version of torchvision.datasets.MNIST.

    Use this class by either passing a path to a torch file (.pt)
    containing (data, targets) or pass the data, targets directly
    instead.
    """

    def __init__(
        self,
        path_to_data=None,
        data=None,
        targets=None,
        transform: Callable | None = None,
    ) -> None:
        path = path_to_data.parent if path_to_data else None
        super(TorchVision_FL, self).__init__(path, transform=transform)
        self.transform = transform

        if path_to_data:
            # load data and targets (path_to_data points to an specific .pt file)
            self.data, self.targets = torch.load(path_to_data)
        else:
            self.data = data
            self.targets = targets

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        if not isinstance(img, Image.Image):  # if not PIL image
            if not isinstance(img, np.ndarray):  # if torch tensor
                img = img.numpy()

            img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)


def get_dataset(path_to_data: Path, cid: str, partition: str):
    # generate path to cid's data
    path_to_data = path_to_data / cid / (partition + ".pt")

    return TorchVision_FL(path_to_data, transform=cifar100Transformation())


def get_dataloader_generators(
    partition_dir: Path,
) -> tuple[ClientDataloaderGen, FedDataloaderGen]:
    """Return a function that loads a client's dataset.

    Parameters
    ----------
    partition_dir : Path
        The path to the partition directory.
        Containing the training data of clients.
        Partitioned by client id.

    Returns
    -------
    Tuple[ClientDataloaderGen, FedDataloaderGen]
        A tuple of functions that return a DataLoader for a client's dataset
        and a DataLoader for the federated dataset.
    """

    def get_client_dataloader(
        cid: CID, test: bool, _config: dict, rng_tuple: IsolatedRNG
    ) -> DataLoader:
        """Return a DataLoader for a client's dataset.

        Parameters
        ----------
        cid : str|int
            The client's ID
        test : bool
            Whether to load the test set or not
        _config : Dict
            The configuration for the dataset
        rng_tuple : IsolatedRNGTuple
            The random number generator state for the training.
            Use if you need seeded random behavior

        Returns
        -------
        DataLoader
            The DataLoader for the client's dataset
        """
        config: ClientDataloaderConfig = ClientDataloaderConfig(**_config)
        del _config

        torch_cpu_generator = rng_tuple[3]

        # client_dir = partition_dir / f"client_{cid}"
        if not test:
            dataset = get_dataset(
                path_to_data=partition_dir, cid=f"client_{cid}", partition="train"
            )
            # dataset = torch.load(client_dir / "train.pt")
        else:
            dataset = get_dataset(
                path_to_data=partition_dir, cid=f"client_{cid}", partition="test"
            )
            # dataset = torch.load(client_dir / "test.pt")
        return DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=not test,
            generator=torch_cpu_generator,
        )

    def get_federated_dataloader(
        test: bool, _config: dict, rng_tuple: IsolatedRNG
    ) -> DataLoader:
        """Return a DataLoader for federated train/test sets.

        Parameters
        ----------
        test : bool
            Whether to load the test set or not
        config : Dict
            The configuration for the dataset
        rng_tuple : IsolatedRNGTuple
            The random number generator state for the training.
            Use if you need seeded random behavior

        Returns
        -------
            DataLoader
            The DataLoader for the federated dataset
        """
        config: FedDataloaderConfig = FedDataloaderConfig(
            **_config,
        )
        del _config
        torch_cpu_generator = rng_tuple[3]

        if not test:
            return DataLoader(
                torch.load(partition_dir / "train.pt"),
                batch_size=config.batch_size,
                shuffle=not test,
                generator=torch_cpu_generator,
            )

        return DataLoader(
            torch.load(partition_dir / "test.pt"),
            batch_size=config.batch_size,
            shuffle=not test,
            generator=torch_cpu_generator,
        )

    return get_client_dataloader, get_federated_dataloader

"""FEMNIST dataset utilities for federated learning."""

# flake8: noqa

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

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

import csv
from pathlib import Path
from typing import Any
from collections.abc import Callable, Sequence
import torch
from PIL import Image
from PIL.Image import Image as ImageType
from torch.utils.data import Dataset

# Use defaults for this very simple dataset
# Requires only batch size
ClientDataloaderConfig = DefaultClientDataloaderConfig
FedDataloaderConfig = DefaultFedDataloaderConfig


class FEMNIST(Dataset):
    """Class to load the FEMNIST dataset."""

    def __init__(
        self,
        mapping: Path,
        data_dir: Path,
        name: str = "train",
        transform: Callable[[ImageType], Any] | None = None,
        target_transform: Callable[[int], Any] | None = None,
    ) -> None:
        """Initialize the FEMNIST dataset.

        Args:
            mapping (Path): path to the mapping folder containing the .csv files.
            data_dir (Path): path to the dataset folder. Defaults to data_dir.
            name (str): name of the dataset to load, train or test.
            transform (Optional[Callable[[ImageType], Any]], optional):
                    transform function to be applied to the ImageType object.
            target_transform (Optional[Callable[[int], Any]], optional):
                    transform function to be applied to the label.
        """
        self.data_dir = data_dir
        self.mapping = mapping
        self.name = name

        self.data: Sequence[tuple[str, int]] = self._load_dataset()
        self.transform: Callable[[ImageType], Any] | None = transform
        self.target_transform: Callable[[int], Any] | None = target_transform

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        """Get a sample.

        Args:
            index (_type_): index of the sample.

        Returns
        -------
            Tuple[Any, Any]: couple (sample, label).
        """
        sample_path, label = self.data[index]

        # Convert to the full path
        full_sample_path: Path = self.data_dir / self.name / sample_path

        img: ImageType = Image.open(full_sample_path).convert("L")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label

    def __len__(self) -> int:
        """Get the length of the dataset as number of samples.

        Returns
        -------
            int: the length of the dataset.
        """
        return len(self.data)

    def _load_dataset(self) -> Sequence[tuple[str, int]]:
        """Load the paths and labels of the partition.

        Preprocess the dataset for faster future loading
        If opened for the first time

        Raises
        ------
            ValueError: raised if the mapping file doesn't exists

        Returns
        -------
            Sequence[Tuple[str, int]]:
                partition asked as a sequence of couples (path_to_file, label)
        """
        preprocessed_path: Path = (self.mapping / self.name).with_suffix(".pt")
        if preprocessed_path.exists():
            return torch.load(preprocessed_path)
        else:
            csv_path = (self.mapping / self.name).with_suffix(".csv")
            if not csv_path.exists():
                raise ValueError(f"Required files do not exist, path: {csv_path}")

            with open(csv_path) as csv_file:
                csv_reader = csv.reader(csv_file)
                # Ignore header
                next(csv_reader)

                # Extract the samples and the labels
                partition: Sequence[tuple[str, int]] = [
                    (sample_path, int(label_id))
                    for _, sample_path, _, label_id in csv_reader
                ]

                # Save for future loading
                torch.save(partition, preprocessed_path)
                return partition


# Load with appropriate transforms
def to_tensor_transform(p: Any) -> torch.Tensor:
    """Transform the object given to a PyTorch Tensor.

    Args:
        p (Any): object to transform

    Returns
    -------
        torch.Tensor: resulting PyTorch Tensor
    """
    return torch.tensor(p)


def load_FEMNIST_dataset(  # noqa: N802
    data_dir: Path, mapping: Path, name: str
) -> Dataset:
    """Load the FEMNIST dataset given the mapping .csv file.

    The relevant transforms are automatically applied.

    Args:
        data_dir (Path): path to the dataset folder.
        mapping (Path): path to the mapping .csv file chosen.
        name (str): name of the dataset to load, train or test.

    Returns
    -------
        Dataset: FEMNIST dataset object, ready-to-use.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    return FEMNIST(
        mapping=mapping,
        name=name,
        data_dir=data_dir,
        transform=transform,
        target_transform=to_tensor_transform,
    )


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

    dataset_dir: Path = partition_dir / "femnist"
    data_dir: Path = dataset_dir / "data"
    centralized_partition: Path = dataset_dir / "client_data_mappings" / "centralized"
    centralized_mapping: Path = (
        dataset_dir / "client_data_mappings" / "centralized" / "0"
    )
    federated_partition: Path = dataset_dir / "client_data_mappings" / "fed_natural"

    def get_client_dataloader(
        cid: CID, test: bool, _config: dict, rng_tuple: IsolatedRNG
    ) -> DataLoader:
        config: ClientDataloaderConfig = ClientDataloaderConfig(**_config)
        del _config

        torch_cpu_generator = rng_tuple[3]

        # data/femnist/partition/femnist/client_data_mappings/fed_natural/0/train.pt
        if not test:
            dataset = load_FEMNIST_dataset(
                data_dir=data_dir,
                mapping=federated_partition / str(cid),
                name="train",
            )
        else:
            dataset = load_FEMNIST_dataset(
                data_dir=data_dir,
                mapping=federated_partition / str(cid),
                name="test",
            )
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
            dataset = load_FEMNIST_dataset(
                data_dir=data_dir,
                mapping=centralized_mapping,
                name="train",
            )
        else:
            dataset = load_FEMNIST_dataset(
                data_dir=data_dir,
                mapping=centralized_mapping,
                name="val",
            )

        return DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=not test,
            generator=torch_cpu_generator,
        )

    return get_client_dataloader, get_federated_dataloader

"""Handle the dataset partitioning and (optionally) complex downloads.

Please add here all the necessary logic to either download, uncompress, pre/post-process
your dataset (or all of the above). If the desired way of running your code is to first
download the dataset and partition it and then run the experiments, please uncomment the
lines below and tell us in the README.md (see the "Running the Experiment" block) that
this file should be executed first.
"""

# flake8: noqa

import hydra
from omegaconf import DictConfig, OmegaConf
from flwr.common.logger import log
import logging

from pathlib import Path
from torchvision import datasets
from torchvision import transforms
import torch
import numpy as np
import shutil

from project.task.cifar10_classification.utils import create_lda_partitions


def cifar10Transformation():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])


def get_cifar_10(path_to_data):
    """Downloads CIFAR10 dataset and generates a unified training set (it will
    be partitioned later using the LDA partitioning mechanism.
    """
    # download dataset and load train set
    train_set = datasets.CIFAR10(root=path_to_data, train=True, download=True)

    # fuse all data splits into a single "training.pt"
    data_loc = Path(path_to_data)
    training_data = data_loc / "training.pt"
    log(logging.INFO, "Generating unified CIFAR dataset")

    torch.save([train_set.data, np.array(train_set.targets)], training_data)

    test_set = datasets.CIFAR10(
        root=path_to_data, train=False, download=True, transform=cifar10Transformation()
    )

    # returns path where training data is and testset
    return training_data, test_set


def get_random_id_splits(total: int, val_ratio: float, shuffle: bool = True):
    """Splits a list of length `total` into two following a
    (1-val_ratio):val_ratio partitioning.

    By default the indices are shuffled before creating the split and
    returning.
    """
    if isinstance(total, int):
        indices = list(range(total))
    else:
        indices = total

    split = int(np.floor(val_ratio * len(indices)))
    if shuffle:
        np.random.shuffle(indices)
    return indices[split:], indices[:split]


def do_fl_partitioning(
    path_to_dataset, partition_dir, pool_size, alpha, num_classes, val_ratio=0.0
):
    """Torchvision (e.g. CIFAR-10) datasets using LDA."""
    images, labels = torch.load(path_to_dataset)
    idx = np.array(range(len(images)))
    dataset = [idx, labels]
    partitions, _ = create_lda_partitions(
        dataset, num_partitions=pool_size, concentration=alpha, accept_imbalanced=True
    )

    # Show label distribution for first partition (purely informative)
    partition_zero = partitions[0][1]
    hist, _ = np.histogram(partition_zero, bins=list(range(num_classes + 1)))
    log(
        logging.INFO,
        f"Class histogram for 0-th partition (alpha={alpha}, {num_classes} classes):"
        f" {hist}",
    )

    # now save partitioned dataset to disk
    # first delete dir containing splits (if exists), then create it
    splits_dir = partition_dir
    if splits_dir.exists():
        shutil.rmtree(splits_dir)
    Path.mkdir(splits_dir, parents=True)

    for p in range(pool_size):
        labels = partitions[p][1]
        image_idx = partitions[p][0]
        imgs = images[image_idx]

        # create dir
        Path.mkdir(splits_dir / f"client_{p}")

        if val_ratio > 0.0:
            # split data according to val_ratio
            train_idx, val_idx = get_random_id_splits(len(labels), val_ratio)
            val_imgs = imgs[val_idx]
            val_labels = labels[val_idx]

            with open(splits_dir / f"client_{p}" / "test.pt", "wb") as f:
                torch.save([val_imgs, val_labels], f)

            # remaining images for training
            imgs = imgs[train_idx]
            labels = labels[train_idx]

        with open(splits_dir / f"client_{p}" / "train.pt", "wb") as f:
            torch.save([imgs, labels], f)

    return splits_dir


@hydra.main(config_path="../../conf", config_name="cifar10", version_base=None)
def download_and_preprocess(cfg: DictConfig) -> None:
    """Download and preprocess the dataset.

    Please include here all the logic
    Please use the Hydra config style as much as possible specially
    for parts that can be customized (e.g. how data is partitioned)

    Parameters
    ----------
    cfg : DictConfig
        An omegaconf object that stores the hydra config.
    """
    # 1. print parsed config
    log(logging.INFO, OmegaConf.to_yaml(cfg))

    # Download the CIFAR-10 dataset
    train_path, testset = get_cifar_10(path_to_data=cfg.dataset.dataset_dir)

    client_pool_size = cfg.dataset.num_clients
    iid_alpha = cfg.dataset.alpha
    val_ratio = cfg.dataset.val_ratio
    partition_dir = Path(cfg.dataset.partition_dir)

    _ = do_fl_partitioning(
        train_path,
        partition_dir=partition_dir,
        pool_size=client_pool_size,
        alpha=iid_alpha,
        num_classes=10,
        val_ratio=val_ratio,
    )
    torch.save(testset, partition_dir / "test.pt")

    log(
        logging.INFO,
        f"Data partitioned across {client_pool_size} clients, with IID alpha ="
        f" {iid_alpha} and {val_ratio} of local dataset reserved for validation.",
    )


if __name__ == "__main__":
    download_and_preprocess()

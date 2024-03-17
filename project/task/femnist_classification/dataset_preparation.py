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
import gdown
import subprocess


def get_femnist(path_to_data, partition_dir):
    """Downloads FEMNIST dataset and generates a unified training set (it will
    be partitioned later using the LDA partitioning mechanism.
    """
    # download dataset and load train set
    path_to_data = Path(path_to_data)
    path_to_data.mkdir(parents=True, exist_ok=True)

    dataset_zip_path = path_to_data / "femnist.tar.gz"
    dataset_dir = path_to_data / "partition"

    #  Download compressed dataset
    if not dataset_zip_path.exists():
        id = "1-CI6-QoEmGiInV23-n_l6Yd8QGWerw8-"
        gdown.download(
            f"https://drive.google.com/uc?export=download&confirm=pbef&id={id}",
            str(dataset_zip_path),
        )

    partition_dir = Path(partition_dir)
    if not partition_dir.exists():
        partition_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(["tar", "-xf", str(dataset_zip_path), "-C", str(partition_dir)])
        print(f"Dataset extracted in {partition_dir}")

    # return training_data, test_set


@hydra.main(config_path="../../conf", config_name="femnist", version_base=None)
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
    print(OmegaConf.to_yaml(cfg))
    log(logging.INFO, OmegaConf.to_yaml(cfg))

    # Download the FEMNIST dataset
    get_femnist(
        path_to_data=cfg.dataset.dataset_dir, partition_dir=cfg.dataset.partition_dir
    )


if __name__ == "__main__":
    download_and_preprocess()

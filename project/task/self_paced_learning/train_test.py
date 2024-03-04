"""CIFAR10 training and testing functions, local and federated."""

from collections.abc import Sized
from pathlib import Path
from typing import cast

import torch
from pydantic import BaseModel
from torch import nn
from torch.utils.data import DataLoader

from project.task.default.train_test import get_fed_eval_fn as get_default_fed_eval_fn
from project.task.default.train_test import (
    get_on_evaluate_config_fn as get_default_on_evaluate_config_fn,
)
from project.task.default.train_test import (
    get_on_fit_config_fn as get_default_on_fit_config_fn,
)
from project.types.common import IsolatedRNG


class TrainConfig(BaseModel):
    """Training configuration, allows '.' member access and static checking.

    Guarantees that all necessary components are present, fails early if config is
    mismatched to client.
    """

    device: torch.device
    epochs: int
    learning_rate: float
    percentage: float | None

    class Config:
        """Setting to allow any types, including library ones like torch.device."""

        arbitrary_types_allowed = True


# this can be implemented in a more efficient way to avoid every instances from being parsed twice
def get_loss_threshold(
    net: nn.Module,
    trainloader: DataLoader,
    percentage: float | None,
    device: torch.device,
) -> float:
    """Get the loss threshold corresponding to the specific percentage for the self-paced learning.

    Parameters
    ----------
    net : nn.Module
        The neural network to train.
    trainloader : DataLoader
        The DataLoader containing the data to train the network on.
    criterion : nn.Module
        The loss function to use.
    percentage : float
        The percentage of the data to use for the self-paced learning.

    Returns
    -------
    nn.Module
        The loss function to use for the self-paced learning.
    """
    net.to(device)
    net.eval()

    criterion = nn.CrossEntropyLoss(reduction="none")
    loss_ls = []

    with torch.no_grad():
        for images, labels in trainloader:
            images, labels = (
                images.to(
                    device,
                ),
                labels.to(device),
            )
            outputs = net(images)
            losses = (
                criterion(
                    outputs,
                    labels,
                )
                .cpu()
                .numpy()
            ).tolist()
            loss_ls.extend(losses)

    loss_ls.sort()
    print(f"Lenght of loss_ls: {len(loss_ls)}, min: {loss_ls[0]}, max: {loss_ls[-1]}")

    if percentage is None:
        return loss_ls[-1]
    else:
        index = min(len(loss_ls), int(len(loss_ls) * percentage)) - 1
        loss_threshold = loss_ls[index]
        print(f"loss_threshold: {loss_threshold}")
        return loss_threshold


def train(  # pylint: disable=too-many-arguments
    net: nn.Module,
    trainloader: DataLoader,
    _config: dict,
    _working_dir: Path,
    _rng_tuple: IsolatedRNG,
) -> tuple[int, dict]:
    """Train the network on the training set.

    Parameters
    ----------
    net : nn.Module
        The neural network to train.
    trainloader : DataLoader
        The DataLoader containing the data to train the network on.
    _config : Dict
        The configuration for the training.
        Contains the device, number of epochs and learning rate.
        Static type checking is done by the TrainConfig class.
    _working_dir : Path
        The working directory for the training.
        Unused.
    _rng_tuple : IsolatedRNGTuple
        The random number generator state for the training.
        Use if you need seeded random behavior

    Returns
    -------
    Tuple[int, Dict]
        The number of samples used for training,
        the loss, and the accuracy of the input model on the given data.
    """
    if len(cast(Sized, trainloader.dataset)) == 0:
        raise ValueError(
            "Trainloader can't be 0, exiting...",
        )
    config: TrainConfig = TrainConfig(**_config)
    del _config

    loss_threshold = get_loss_threshold(
        net,
        trainloader,
        config.percentage,
        config.device,
    )
    print(f"loss_threshold: {loss_threshold}")

    net.to(config.device)
    net.train()

    criterion = nn.CrossEntropyLoss(reduction="none")
    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=config.learning_rate,
    )
    final_epoch_per_sample_loss = 0.0
    num_correct = 0
    for _ in range(config.epochs):
        final_epoch_per_sample_loss = 0.0
        num_correct = 0
        for data, target in trainloader:
            data, target = (
                data.to(
                    config.device,
                ),
                target.to(config.device),
            )
            optimizer.zero_grad()
            output = net(data)
            losses = criterion(output, target)

            # only learn on samples with loss < loss_threshold
            mask = losses <= loss_threshold
            loss = (
                losses * mask
            ).sum() / mask.sum()  # shouldn't directly use mean() here

            final_epoch_per_sample_loss += loss.item()
            num_correct += (output.max(1)[1] == target).clone().detach().sum().item()
            loss.backward()
            optimizer.step()

    return len(cast(Sized, trainloader.dataset)), {
        "train_loss": final_epoch_per_sample_loss
        / len(cast(Sized, trainloader.dataset)),
        "train_accuracy": float(num_correct) / len(cast(Sized, trainloader.dataset)),
    }


class TestConfig(BaseModel):
    """Testing configuration, allows '.' member access and static checking.

    Guarantees that all necessary components are present, fails early if config is
    mismatched to client.
    """

    device: torch.device

    class Config:
        """Setting to allow any types, including library ones like torch.device."""

        arbitrary_types_allowed = True


def test(
    net: nn.Module,
    testloader: DataLoader,
    _config: dict,
    _working_dir: Path,
    _rng_tuple: IsolatedRNG,
) -> tuple[float, int, dict]:
    """Evaluate the network on the test set.

    Parameters
    ----------
    net : nn.Module
        The neural network to test.
    testloader : DataLoader
        The DataLoader containing the data to test the network on.
    _config : Dict
        The configuration for the testing.
        Contains the device.
        Static type checking is done by the TestConfig class.
    _working_dir : Path
        The working directory for the training.
        Unused.
    _rng_tuple : IsolatedRNGTuple
        The random number generator state for the training.
        Use if you need seeded random behavior


    Returns
    -------
    Tuple[float, int, float]
        The loss, number of test samples,
        and the accuracy of the input model on the given data.
    """
    if len(cast(Sized, testloader.dataset)) == 0:
        raise ValueError(
            "Testloader can't be 0, exiting...",
        )

    config: TestConfig = TestConfig(**_config)
    del _config

    net.to(config.device)
    net.eval()

    criterion = nn.CrossEntropyLoss()
    correct, per_sample_loss = 0, 0.0

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = (
                images.to(
                    config.device,
                ),
                labels.to(config.device),
            )
            outputs = net(images)
            per_sample_loss += criterion(
                outputs,
                labels,
            ).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

    return (
        per_sample_loss / len(cast(Sized, testloader.dataset)),
        len(cast(Sized, testloader.dataset)),
        {
            "test_accuracy": float(correct) / len(cast(Sized, testloader.dataset)),
        },
    )


# Use defaults as they are completely determined
# by the other functions defined in mnist_classification
get_fed_eval_fn = get_default_fed_eval_fn
get_on_fit_config_fn = get_default_on_fit_config_fn
get_on_evaluate_config_fn = get_default_on_evaluate_config_fn

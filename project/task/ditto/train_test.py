"""CIFAR10 training and testing functions, local and federated."""

from collections.abc import Sized
from pathlib import Path
from typing import cast

import torch
import copy
from pydantic import BaseModel
from torch import nn
from torch.utils.data import DataLoader

from project.task.self_paced_learning.pacing_utils import (
    get_filtered_trainloader,
)

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
    momentum: float
    percentage: float | None
    is_anti: bool | None
    is_random: bool | None

    class Config:
        """Setting to allow any types, including library ones like torch.device."""

        arbitrary_types_allowed = True


def train(  # pylint: disable=too-many-arguments
    net: nn.Module,
    local_net: nn.Module,
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

    net.to(config.device)
    local_net.to(config.device)

    frozen_teacher_net = copy.deepcopy(net)
    frozen_teacher_net.to(config.device)
    for param in frozen_teacher_net.parameters():
        param.requires_grad = False

    trainloader_filtered = get_filtered_trainloader(
        frozen_teacher_net,
        trainloader,
        config.percentage,
        config.device,
        ascending_order=not config.is_anti,
        is_random=config.is_random if config.is_random is not None else False,
    )
    print(
        "Length of trainloader after filtering:"
        f" {len(cast(Sized, trainloader_filtered.dataset))}"
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        net.parameters(), lr=config.learning_rate, momentum=config.momentum
    )
    optimiser_local = torch.optim.SGD(
        local_net.parameters(), lr=config.learning_rate, momentum=config.momentum
    )

    frac = 10  # trade-off between CrossEntropy loss and knowledge distillation
    final_epoch_per_sample_loss = 0.0
    num_correct = 0
    final_epoch_per_sample_loss_local = 0.0
    num_correct_local = 0
    for i in range(config.epochs):
        net.train()
        final_epoch_per_sample_loss = 0.0
        num_correct = 0
        for data, target in trainloader_filtered:
            data, target = (
                data.to(
                    config.device,
                ),
                target.to(config.device),
            )

            output = net(data)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            final_epoch_per_sample_loss += loss.item()
            num_correct += (output.max(1)[1] == target).clone().detach().sum().item()

        print(
            f"Epoch {i + 1}, loss:"
            f" {final_epoch_per_sample_loss / len(trainloader_filtered.dataset)},"
            f" accuracy: {num_correct / len(trainloader_filtered.dataset)}, local loss"
        )
    for i in range(config.epochs):
        local_net.train()
        net.eval()
        final_epoch_per_sample_loss_local = 0.0
        num_correct_local = 0
        for data, target in trainloader_filtered:
            data, target = (
                data.to(
                    config.device,
                ),
                target.to(config.device),
            )
        output_local = local_net(data)
        w_local = torch.hstack([param.flatten() for param in local_net.parameters()])
        w_global_frozen = torch.hstack(
            [param.flatten() for param in copy.deepcopy(net).parameters()]
        )
        ditto_sim = torch.sum((w_local - w_global_frozen) ** 2)
        loss_local = criterion(output_local, target) + frac * ditto_sim / 2.0
        print(f"-----------------Ditto: {ditto_sim.item()}--------------------")
        optimiser_local.zero_grad()
        loss_local.backward()
        optimiser_local.step()
        final_epoch_per_sample_loss_local += loss_local.item()
        num_correct_local += (
            (output_local.max(1)[1] == target).clone().detach().sum().item()
        )
        print(
            f"Epoch {i + 1}, loss:"
            f" {final_epoch_per_sample_loss_local / len(trainloader_filtered.dataset)},"
            f" local accuracy: {num_correct_local / len(trainloader_filtered.dataset)}"
        )

    return len(cast(Sized, trainloader_filtered.dataset)), {
        "train_loss": final_epoch_per_sample_loss
        / len(cast(Sized, trainloader_filtered.dataset)),
        "train_accuracy": float(num_correct)
        / len(cast(Sized, trainloader_filtered.dataset)),
        "local_train_loss": final_epoch_per_sample_loss_local
        / len(cast(Sized, trainloader_filtered.dataset)),
        "local_train_accuracy": float(num_correct_local)
        / len(cast(Sized, trainloader_filtered.dataset)),
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

            # evaluate federated model
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

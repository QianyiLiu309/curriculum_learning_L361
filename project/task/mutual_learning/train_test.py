"""CIFAR10 training and testing functions, local and federated."""

from collections.abc import Sized
from pathlib import Path
from typing import cast

import torch
import copy
from pydantic import BaseModel
from torch import nn
from torch.utils.data import DataLoader

from project.task.self_paced_learning.pacing_utils import get_loss_threshold

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

    net.to(config.device)
    personal_net = copy.deepcopy(net)
    personal_net.to(config.device)

    frozen_teacher_net = copy.deepcopy(net)
    frozen_teacher_net.to(config.device)
    for param in frozen_teacher_net.parameters():
        param.requires_grad = False

    loss_threshold = get_loss_threshold(
        frozen_teacher_net,
        trainloader,
        config.percentage,
        config.device,
    )
    print(f"loss_threshold based on frozen teacher: {loss_threshold}")

    criterion = nn.CrossEntropyLoss(reduction="none")
    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=config.learning_rate,
    )
    optimizer_personal = torch.optim.SGD(
        personal_net.parameters(),
        lr=config.learning_rate
    )
    final_epoch_per_sample_loss = 0.0
    num_correct = 0
    for i in range(config.epochs):
        net.train()
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
            output_personal = personal_net(data)
            DL = 0.5 * (torch.sum(output_personal * torch.log(output_personal / output), dim=1) + 
                        torch.sum(output * torch.log(output / output_personal), dim=1))
            losses = criterion(output, target) + 0.001 * DL
            # TODO: check 
            #           1. net_personal 
            #           2. set a trade-off param lambda
            #           3. dimension of KL
            losses_personal = losses + 0.001 * DL

            with torch.no_grad():
                teacher_output = frozen_teacher_net(data)
                teacher_losses = criterion(teacher_output, target)

            # only learn on samples with loss < loss_threshold
            mask = teacher_losses <= loss_threshold
            loss = (losses * mask).sum() / (
                mask.sum() + 1e-10
            )  # shouldn't directly use mean() here
            loss_personal = (losses_personal * mask).sum() / (
                mask.sum() + 1e-10
            )

            final_epoch_per_sample_loss += loss.item()
            num_correct += (output.max(1)[1] == target).clone().detach().sum().item()
            loss.backward()
            optimizer.step()

            loss_personal.backward()
            optimizer_personal.step()
        print(
            f"Epoch {i + 1}, loss:"
            f" {final_epoch_per_sample_loss / len(trainloader.dataset)},"
            f" accuracy: {num_correct / len(trainloader.dataset)}"
        )

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
    personal_net.to(config.device)
    personal_net.eval()

    criterion = nn.CrossEntropyLoss()
    correct, per_sample_loss = 0, 0.0
    correct_personal, per_sample_loss_personal = 0, 0.0

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = (
                images.to(
                    config.device,
                ),
                labels.to(config.device),
            )
            outputs = net(images)
            outputs_personal = personal_net(images)
            loss = criterion(
                outputs,
                labels,
            ).item()
            per_sample_loss += loss
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

            per_sample_loss_personal += loss + torch.sum(outputs_personal * torch.log(outputs_personal * outputs)).item()
            _, predicted_personal = torch.max(outputs_personal.data, 1)
            correct_personal += (predicted_personal == labels).sum().item()

    return (
        per_sample_loss / len(cast(Sized, testloader.dataset)),
        per_sample_loss_personal / len(cast(Sized, testloader.dataset)),
        len(cast(Sized, testloader.dataset)),
        {
            "test_accuracy": float(correct) / len(cast(Sized, testloader.dataset)),
            "test_accuracy_personal": float(correct_personal) / len(cast(Sized, testloader.dataset))
        },
    )


# Use defaults as they are completely determined
# by the other functions defined in mnist_classification
get_fed_eval_fn = get_default_fed_eval_fn
get_on_fit_config_fn = get_default_on_fit_config_fn
get_on_evaluate_config_fn = get_default_on_evaluate_config_fn

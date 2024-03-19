"""Helper functions for curriculum learning."""

import torch
from torch import nn
from torch.utils.data import DataLoader, default_collate

import random


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
        index = max(min(len(loss_ls), int(len(loss_ls) * percentage)) - 1, 0)
        loss_threshold = loss_ls[index]
        print(f"loss_threshold: {loss_threshold}")
        return loss_threshold


def get_filtered_trainloader(
    net: nn.Module,
    trainloader: DataLoader,
    percentage: float | None,
    device: torch.device,
    ascending_order: bool = True,
    is_random: bool = False,
) -> DataLoader:
    """Get the filtered trainloader corresponding to the specific percentage for curriculum learning.

    Parameters
    ----------
    net : nn.Module
        The neural network to train.
    trainloader : DataLoader
        The DataLoader containing the data to train the network on.
    criterion : nn.Module
        The loss function to use.
    percentage : float
        The percentage of the data to use for curriculum learning.

    Returns
    -------
    nn.Module
        The loss threshold to use for curriculum learning.
    """
    if percentage is None:
        return trainloader

    print(f"Percentage: {percentage}")

    net.to(device)
    net.eval()

    criterion = nn.CrossEntropyLoss(reduction="none")
    loss_ls = []

    with torch.no_grad():
        for i in range(len(trainloader.dataset)):
            images, labels = trainloader.dataset[i]
            images, labels = default_collate([(images, labels)])
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss = criterion(outputs, labels).item()
            loss_ls.append(loss)

    sorted_loss_ls = sorted(loss_ls, reverse=not ascending_order)
    print(
        f"Lenght of loss_ls: {len(sorted_loss_ls)}, left bound: {sorted_loss_ls[0]},"
        f" right bound: {sorted_loss_ls[-1]}"
    )

    index = max(min(len(sorted_loss_ls), int(len(sorted_loss_ls) * percentage)) - 1, 0)
    print(f"index: {index}")
    loss_threshold = sorted_loss_ls[index]
    print(f"loss_threshold: {loss_threshold}")

    if is_random:
        print("Randomly sampling the data.")
        full_indices = list(range(len(trainloader.dataset)))
        filtered_indices = random.sample(full_indices, index + 1)
    elif ascending_order:
        filtered_indices = [
            i for i, loss in enumerate(loss_ls) if loss <= loss_threshold
        ]
    else:
        filtered_indices = [
            i for i, loss in enumerate(loss_ls) if loss >= loss_threshold
        ]
    print(f"Number of samples before filtering: {len(trainloader.dataset)}")
    print(f"Number of indices after filtering: {len(filtered_indices)}")
    filtered_dataest = torch.utils.data.Subset(trainloader.dataset, filtered_indices)
    print(f"Number of samples after filtering: {len(filtered_dataest)}")
    filtered_trainloader = torch.utils.data.DataLoader(
        filtered_dataest,
        batch_size=trainloader.batch_size,
        shuffle=True,
        generator=trainloader.generator,
    )
    return filtered_trainloader

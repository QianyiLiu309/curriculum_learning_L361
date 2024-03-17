"""Helper functions for curriculum learning."""

import torch
from torch import nn
from torch.utils.data import DataLoader


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

"""ResNet model implementation."""

import torch
from torch import nn

from project.types.common import IsolatedRNG


class Block(nn.Module):
    """Block class."""

    def __init__(
        self,
        num_layers: int,
        in_channels: int,
        out_channels: int,
        identity_downsample: nn.Module | None = None,
        stride: int = 1,
    ) -> None:
        assert num_layers in {18, 34, 50, 101, 152}, "should be a a valid architecture"
        super().__init__()
        self.num_layers = num_layers
        if self.num_layers > 34:
            self.expansion = 4
        else:
            self.expansion = 1
        # ResNet50, 101, and 152 include additional layer of 1x1 kernels
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0
        )
        if self.num_layers > 34:
            self.bn1 = nn.BatchNorm2d(out_channels)
        if self.num_layers > 34:
            self.conv2 = nn.Conv2d(
                out_channels, out_channels, kernel_size=3, stride=stride, padding=1
            )
        else:
            # for ResNet18 and 34, connect input directly
            # to (3x3) kernel (skip first (1x1))
            self.conv2 = nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=stride, padding=1
            )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(
            out_channels,
            out_channels * self.expansion,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        if self.num_layers > 34:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    """ResNet class."""

    def __init__(self, num_layers: int, image_channels: int, num_classes: int) -> None:
        assert num_layers in {18, 34, 50, 101, 152}, (
            f"ResNet{num_layers}: Unknown architecture! Number of layers has "
            "to be 18, 34, 50, 101, or 152 "
        )
        super().__init__()
        if num_layers < 50:
            self.expansion = 1
        else:
            self.expansion = 4
        if num_layers == 18:
            layers = [2, 2, 2, 2]
        elif num_layers in {34, 50}:
            layers = [3, 4, 6, 3]
        elif num_layers == 101:
            layers = [3, 4, 23, 3]
        else:
            layers = [3, 8, 36, 3]
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNetLayers
        self.layer1 = self.make_layers(
            num_layers, layers[0], intermediate_channels=64, stride=1
        )
        self.layer2 = self.make_layers(
            num_layers, layers[1], intermediate_channels=128, stride=2
        )
        self.layer3 = self.make_layers(
            num_layers, layers[2], intermediate_channels=256, stride=2
        )
        self.layer4 = self.make_layers(
            num_layers, layers[3], intermediate_channels=512, stride=2
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * self.expansion, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

    def make_layers(
        self,
        num_layers: int,
        num_residual_blocks: int,
        intermediate_channels: int,
        stride: int,
    ) -> nn.Sequential:
        layers = []

        identity_downsample = nn.Sequential(
            nn.Conv2d(
                self.in_channels,
                intermediate_channels * self.expansion,
                kernel_size=1,
                stride=stride,
            ),
            nn.BatchNorm2d(intermediate_channels * self.expansion),
        )
        layers.append(
            Block(
                num_layers,
                self.in_channels,
                intermediate_channels,
                identity_downsample,
                stride,
            )
        )
        self.in_channels = intermediate_channels * self.expansion  # 256
        for _ in range(num_residual_blocks - 1):
            layers.append(
                Block(num_layers, self.in_channels, intermediate_channels)
            )  # 256 -> 64, 64*4 (256) again
        return nn.Sequential(*layers)


def get_resnet(config: dict, _rng_tuple: IsolatedRNG) -> nn.Module:
    """Return a ResNet model with the given configuration.

    Parameters
    ----------
    config : dict
        The configuration for the model.
    seed : IsolatedRNG
        The random number generator.

    Returns
    -------
    nn.Module
        The ResNet model.
    """
    model = ResNet(
        num_layers=config["num_layers"],
        image_channels=config["image_channels"],
        num_classes=config["num_classes"],
    )
    if config.get("initial_run"):
        _ = model(torch.rand(8, config["image_channels"], 32, 32))
    return model

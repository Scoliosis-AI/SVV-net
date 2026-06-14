import torch
from torch import nn


class PatchDiscriminator(nn.Module):
    def __init__(
        self,
        image_channels: int = 1,
        base_channels: int = 64,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        layers = [
            nn.Conv2d(image_channels, base_channels, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        multiplier = 1
        for level in range(1, num_layers + 1):
            previous = multiplier
            multiplier = min(2 ** level, 8)
            layers.extend(
                [
                    nn.Conv2d(
                        base_channels * previous,
                        base_channels * multiplier,
                        4,
                        2 if level < num_layers else 1,
                        1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(base_channels * multiplier),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
        layers.append(nn.Conv2d(base_channels * multiplier, 1, 4, 1, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images)

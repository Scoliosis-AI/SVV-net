from typing import Tuple

import torch
from torch import nn

from .config import ModelConfig
from .modules import (
    AttentionBlock,
    Downsample,
    ResidualBlock,
    Swish,
    Upsample,
    group_norm,
)
from .quantizer import VectorQuantizer


class Encoder(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        channels = config.channels
        layers = [nn.Conv2d(config.image_channels, channels[0], 3, padding=1)]
        resolution = config.base_resolution

        for level, out_channels in enumerate(channels[1:]):
            in_channels = channels[level]
            for _ in range(config.num_res_blocks):
                layers.append(ResidualBlock(in_channels, out_channels))
                in_channels = out_channels
                if resolution in config.attention_resolutions:
                    layers.append(AttentionBlock(in_channels))
            if level != len(channels) - 2:
                layers.append(Downsample(out_channels))
                resolution //= 2

        layers.extend(
            [
                ResidualBlock(channels[-1], channels[-1]),
                AttentionBlock(channels[-1]),
                ResidualBlock(channels[-1], channels[-1]),
                group_norm(channels[-1]),
                Swish(),
                nn.Conv2d(channels[-1], config.latent_dim, 3, padding=1),
            ]
        )
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class Decoder(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        channels = config.decoder_channels
        in_channels = config.channels[-1]
        resolution = config.base_resolution // (2 ** (len(config.channels) - 2))
        layers = [
            nn.Conv2d(config.latent_dim, in_channels, 3, padding=1),
            ResidualBlock(in_channels, in_channels),
            AttentionBlock(in_channels),
            ResidualBlock(in_channels, in_channels),
        ]

        for level, out_channels in enumerate(channels):
            for _ in range(config.num_res_blocks + 1):
                layers.append(ResidualBlock(in_channels, out_channels))
                in_channels = out_channels
                if resolution in config.attention_resolutions:
                    layers.append(AttentionBlock(in_channels))
            if level != 0:
                layers.append(Upsample(in_channels))
                resolution *= 2

        layers.extend(
            [
                group_norm(in_channels),
                Swish(),
                nn.Conv2d(in_channels, config.image_channels, 3, padding=1),
            ]
        )
        self.model = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.model(z)


class VQVAE(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.quant_conv = nn.Conv2d(config.latent_dim, config.latent_dim, 1)
        self.post_quant_conv = nn.Conv2d(
            config.latent_dim, config.latent_dim, 1
        )
        self.quantizer = VectorQuantizer(
            config.num_codebook_vectors,
            config.latent_dim,
            config.commitment_cost,
        )

    def encode(
        self, images: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded = self.quant_conv(self.encoder(images))
        return self.quantizer(encoded)

    def decode(self, quantized: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.post_quant_conv(quantized))

    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        return self.decode(self.quantizer.lookup(indices))

    def forward(
        self, images: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        quantized, indices, quantization_loss = self.encode(images)
        reconstruction = self.decode(quantized)
        return reconstruction, indices, quantization_loss

import torch
from torch import nn
from torch.nn import functional as F


class Swish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


def group_norm(channels: int) -> nn.GroupNorm:
    if channels % 32 != 0:
        raise ValueError(f"channels must be divisible by 32, got {channels}")
    return nn.GroupNorm(32, channels, eps=1e-6, affine=True)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            group_norm(in_channels),
            Swish(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            group_norm(out_channels),
            Swish(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )
        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.skip(x) + self.block(x)


class Downsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (0, 1, 0, 1)))


class Upsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class AttentionBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = group_norm(channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        q, k, v = self.q(h), self.k(h), self.v(h)
        batch, channels, height, width = q.shape

        q = q.reshape(batch, channels, -1).permute(0, 2, 1)
        k = k.reshape(batch, channels, -1)
        weights = torch.bmm(q, k) * (channels ** -0.5)
        weights = torch.softmax(weights, dim=-1)

        v = v.reshape(batch, channels, -1)
        attended = torch.bmm(v, weights.permute(0, 2, 1))
        attended = attended.reshape(batch, channels, height, width)
        return x + self.proj_out(attended)

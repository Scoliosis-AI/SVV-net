from pathlib import Path
from typing import Dict, Optional

import torch
from torch import nn
from torch.optim import Optimizer

from .config import ModelConfig


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optimizer,
    epoch: int,
    config: ModelConfig,
    discriminator: Optional[nn.Module] = None,
    discriminator_optimizer: Optional[Optimizer] = None,
    best_loss: Optional[float] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": config.to_dict(),
    }
    if best_loss is not None:
        checkpoint["best_loss"] = best_loss
    if discriminator is not None:
        checkpoint["discriminator"] = discriminator.state_dict()
    if discriminator_optimizer is not None:
        checkpoint["discriminator_optimizer"] = (
            discriminator_optimizer.state_dict()
        )
    if scaler is not None:
        checkpoint["scaler"] = scaler.state_dict()
    torch.save(checkpoint, destination)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    discriminator: Optional[nn.Module] = None,
    discriminator_optimizer: Optional[Optimizer] = None,
    map_location: str = "cpu",
    scaler: Optional[torch.amp.GradScaler] = None,
) -> int:
    checkpoint: Dict = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if discriminator is not None and "discriminator" in checkpoint:
        discriminator.load_state_dict(checkpoint["discriminator"])
    if (
        discriminator_optimizer is not None
        and "discriminator_optimizer" in checkpoint
    ):
        discriminator_optimizer.load_state_dict(
            checkpoint["discriminator_optimizer"]
        )
    if scaler is not None and "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])
    return int(checkpoint.get("epoch", 0))


def load_model(path: str, device: torch.device) -> nn.Module:
    from .model import VQVAE

    checkpoint: Dict = torch.load(path, map_location=device)
    config = ModelConfig.from_dict(checkpoint["config"])
    model = VQVAE(config).to(device)
    model.load_state_dict(checkpoint["model"])
    return model

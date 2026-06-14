import argparse
from pathlib import Path
from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torchvision.utils import save_image
from tqdm import tqdm

from vqvae import ModelConfig, VQVAE
from vqvae.checkpoint import load_checkpoint, save_checkpoint
from vqvae.data import create_dataloader
from vqvae.discriminator import PatchDiscriminator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the SVV-Net VQ-VAE, optionally with a PatchGAN "
            "discriminator."
        )
    )
    parser.add_argument(
        "--dataset",
        required=True,
        nargs="+",
        help="One or more image folders, e.g. pre-op and post-op folders.",
    )
    parser.add_argument("--output-dir", default="runs/vqvae")
    parser.add_argument("--resume")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--image-channels", type=int, choices=(1, 3), default=1)
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--codebook-size", type=int, default=256)
    parser.add_argument("--commitment-cost", type=float, default=0.25)
    parser.add_argument(
        "--model-preset",
        choices=("highres", "standard", "fast"),
        default="highres",
        help=(
            "'highres' is optimized for 1024x256 images and keeps a 32x8 "
            "latent map; 'standard' and 'fast' retain the older architecture."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use CUDA automatic mixed precision for faster training.",
    )
    parser.add_argument(
        "--channels-last",
        action="store_true",
        help="Use channels-last memory format to accelerate CUDA convolutions.",
    )
    parser.add_argument("--use-gan", action="store_true")
    parser.add_argument("--disc-start", type=int, default=10_000)
    parser.add_argument("--disc-weight", type=float, default=1.0)
    parser.add_argument(
        "--sample-every",
        type=int,
        default=3,
        help="Save a reconstruction preview every N epochs.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=3,
        help="Save latest and numbered checkpoints every N epochs.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def set_requires_grad(module: nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


def adaptive_weight(
    reconstruction_loss: torch.Tensor,
    adversarial_loss: torch.Tensor,
    last_layer: torch.Tensor,
) -> torch.Tensor:
    reconstruction_grad = torch.autograd.grad(
        reconstruction_loss, last_layer, retain_graph=True
    )[0]
    adversarial_grad = torch.autograd.grad(
        adversarial_loss, last_layer, retain_graph=True
    )[0]
    weight = reconstruction_grad.norm() / (adversarial_grad.norm() + 1e-4)
    return (0.8 * weight.clamp(0, 1e4)).detach()


def train_epoch(
    model: VQVAE,
    discriminator: PatchDiscriminator,
    loader: Iterable,
    model_optimizer: Adam,
    discriminator_optimizer: Adam,
    device: torch.device,
    global_step: int,
    use_gan: bool,
    disc_start: int,
    disc_weight: float,
    scaler: torch.amp.GradScaler,
    amp_enabled: bool,
    current_epoch: int,
    total_epochs: int,
    channels_last: bool,
):
    model.train()
    discriminator.train()
    progress = tqdm(
        loader,
        desc=f"epoch {current_epoch}/{total_epochs}",
        leave=False,
    )
    last_batch = None
    total_epoch_loss = 0.0
    num_batches = 0

    for images in progress:
        images = images.to(device, non_blocking=True)
        if channels_last:
            images = images.contiguous(memory_format=torch.channels_last)
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            reconstruction, _, quantization_loss = model(images)
            reconstruction_loss = F.l1_loss(reconstruction, images)

            gan_active = use_gan and global_step >= disc_start
            if gan_active:
                set_requires_grad(discriminator, False)
                generator_loss = -discriminator(reconstruction).mean()
                weight = adaptive_weight(
                    reconstruction_loss,
                    generator_loss,
                    model.decoder.model[-1].weight,
                )
                total_loss = (
                    reconstruction_loss
                    + quantization_loss
                    + disc_weight * weight * generator_loss
                )
            else:
                generator_loss = reconstruction_loss.new_zeros(())
                total_loss = reconstruction_loss + quantization_loss

        model_optimizer.zero_grad(set_to_none=True)
        scaler.scale(total_loss).backward()
        scaler.step(model_optimizer)

        discriminator_loss = reconstruction_loss.new_zeros(())
        if gan_active:
            set_requires_grad(discriminator, True)
            discriminator_optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                real_logits = discriminator(images)
                fake_logits = discriminator(reconstruction.detach())
                discriminator_loss = 0.5 * (
                    F.relu(1.0 - real_logits).mean()
                    + F.relu(1.0 + fake_logits).mean()
                )
            scaler.scale(discriminator_loss).backward()
            scaler.step(discriminator_optimizer)
        scaler.update()

        global_step += 1
        total_epoch_loss += total_loss.detach().item()
        num_batches += 1
        last_batch = (images.detach(), reconstruction.detach())
        progress.set_postfix(
            reconstruction=f"{reconstruction_loss.item():.4f}",
            quantization=f"{quantization_loss.item():.4f}",
            discriminator=f"{discriminator_loss.item():.4f}",
        )

    average_loss = total_epoch_loss / max(num_batches, 1)
    return global_step, last_batch, average_loss


def main() -> None:
    args = parse_args()
    if args.height % 16 or args.width % 16:
        raise ValueError("height and width must both be divisible by 16")
    if args.sample_every < 1 or args.save_every < 1:
        raise ValueError("sample-every and save-every must be positive")
    device = torch.device(args.device)
    amp_enabled = args.amp and device.type == "cuda"
    if args.amp and not amp_enabled:
        print("AMP requested but CUDA is unavailable; continuing without AMP.")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.resume:
        saved = torch.load(args.resume, map_location="cpu")
        config = ModelConfig.from_dict(saved["config"])
    else:
        config_kwargs = {
            "image_channels": args.image_channels,
            "latent_dim": args.latent_dim,
            "num_codebook_vectors": args.codebook_size,
            "commitment_cost": args.commitment_cost,
        }
        if args.model_preset == "highres":
            config_kwargs.update(
                channels=(64, 64, 64, 128, 128, 256, 256),
                decoder_channels=(256, 256, 128, 128, 64, 64),
                num_res_blocks=1,
                attention_resolutions=(),
                base_resolution=1024,
            )
        elif args.model_preset == "fast":
            config_kwargs.update(
                channels=(64, 64, 64, 128, 128, 256),
                decoder_channels=(256, 128, 128, 64, 64),
                num_res_blocks=1,
                attention_resolutions=(),
            )
        config = ModelConfig(**config_kwargs)
    model = VQVAE(config).to(device)
    discriminator = PatchDiscriminator(config.image_channels).to(device)
    if args.channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
        discriminator = discriminator.to(memory_format=torch.channels_last)
    model_optimizer = Adam(model.parameters(), lr=args.learning_rate)
    discriminator_optimizer = Adam(
        discriminator.parameters(), lr=args.learning_rate
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    loader = create_dataloader(
        args.dataset,
        (args.height, args.width),
        config.image_channels,
        args.batch_size,
        args.augment,
        args.num_workers,
    )

    start_epoch = 0
    best_loss = float("inf")
    if args.resume:
        best_loss = float(saved.get("best_loss", float("inf")))
        start_epoch = load_checkpoint(
            args.resume,
            model,
            model_optimizer,
            discriminator,
            discriminator_optimizer,
            map_location=str(device),
            scaler=scaler,
        )

    print(
        f"device={device}, amp={amp_enabled}, batch_size={args.batch_size}, "
        f"epochs={args.epochs}, batches_per_epoch={len(loader)}, "
        f"parameters={sum(p.numel() for p in model.parameters()) / 1e6:.2f}M"
    )
    global_step = start_epoch * len(loader)
    for epoch in range(start_epoch + 1, args.epochs + 1):
        global_step, last_batch, average_loss = train_epoch(
            model,
            discriminator,
            loader,
            model_optimizer,
            discriminator_optimizer,
            device,
            global_step,
            args.use_gan,
            args.disc_start,
            args.disc_weight,
            scaler,
            amp_enabled,
            epoch,
            args.epochs,
            args.channels_last and device.type == "cuda",
        )
        print(
            f"epoch {epoch}/{args.epochs}, "
            f"average_loss={average_loss:.6f}"
        )

        if last_batch and epoch % args.sample_every == 0:
            real, reconstructed = last_batch
            preview = torch.cat([real[:4], reconstructed[:4]]).add(1).div(2)
            save_image(preview, output_dir / f"reconstruction_{epoch:04d}.png")

        if average_loss < best_loss:
            best_loss = average_loss
            save_checkpoint(
                str(output_dir / "best.pt"),
                model,
                model_optimizer,
                epoch,
                config,
                discriminator if args.use_gan else None,
                discriminator_optimizer if args.use_gan else None,
                best_loss,
                scaler,
            )

        if epoch % args.save_every == 0 or epoch == args.epochs:
            save_checkpoint(
                str(output_dir / "last.pt"),
                model,
                model_optimizer,
                epoch,
                config,
                discriminator if args.use_gan else None,
                discriminator_optimizer if args.use_gan else None,
                best_loss,
                scaler,
            )


if __name__ == "__main__":
    main()

import argparse
from pathlib import Path

import torch
from torchvision.utils import save_image
from tqdm import tqdm

from vqvae.checkpoint import load_model
from vqvae.data import create_dataloader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstruct an image folder with the SVV-Net VQ-VAE."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", default="reconstructions")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if args.height % 16 or args.width % 16:
        raise ValueError("height and width must both be divisible by 16")
    device = torch.device(args.device)
    model = load_model(args.checkpoint, device)
    model.eval()
    loader = create_dataloader(
        args.input_dir,
        (args.height, args.width),
        model.config.image_channels,
        args.batch_size,
        augment=False,
        num_workers=0,
        shuffle=False,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_index = 0
    for images in tqdm(loader):
        images = images.to(device)
        reconstructed, _, _ = model(images)
        reconstructed = reconstructed.add(1).div(2).clamp(0, 1)
        for image in reconstructed:
            save_image(image, output_dir / f"{image_index:06d}.png")
            image_index += 1


if __name__ == "__main__":
    main()

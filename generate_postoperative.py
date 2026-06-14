import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

from vqvae.checkpoint import load_model
from vqvae.conditional_transformer import (
    ConditionalCodeTransformer,
    TransformerConfig,
)
from vqvae.paired_data import (
    find_images,
    load_metadata,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate postoperative X-rays with trained SVV-Net checkpoints."
        )
    )
    parser.add_argument("--vqvae-checkpoint", required=True)
    parser.add_argument("--transformer-checkpoint", required=True)
    parser.add_argument("--preoperative-dir", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output-dir", default="generated_postoperative")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    vqvae = load_model(args.vqvae_checkpoint, device)
    vqvae.eval()

    checkpoint = torch.load(args.transformer_checkpoint, map_location=device)
    transformer_config = TransformerConfig.from_dict(checkpoint["config"])
    model = ConditionalCodeTransformer(transformer_config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    latent_shape = tuple(checkpoint["latent_shape"])

    metadata = load_metadata(args.metadata)
    images = find_images(args.preoperative_dir)
    case_ids = sorted(set(images) & set(metadata))
    if not case_ids:
        raise ValueError("no preoperative images match the metadata table")

    channels = vqvae.config.image_channels
    mode = "L" if channels == 1 else "RGB"
    image_transform = transforms.Compose(
        [
            transforms.Resize((args.height, args.width)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * channels, [0.5] * channels),
        ]
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for case_id in tqdm(case_ids):
        with Image.open(images[case_id]) as image:
            preoperative = image_transform(image.convert(mode))
        preoperative = preoperative.unsqueeze(0).to(device)
        _, indices, _ = vqvae.encode(preoperative)
        pre_tokens = indices.flatten(1)
        uiv, liv = metadata[case_id]
        generated = model.generate(
            pre_tokens,
            torch.tensor([uiv], device=device),
            torch.tensor([liv], device=device),
            temperature=args.temperature,
            top_k=args.top_k,
            sample=args.sample,
        )
        generated = generated.view(1, *latent_shape)
        postoperative = vqvae.decode_indices(generated)
        save_image(
            postoperative[0].add(1).div(2).clamp(0, 1),
            output_dir / f"{case_id}.png",
        )


if __name__ == "__main__":
    main()

import argparse
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from vqvae.checkpoint import load_model
from vqvae.conditional_transformer import (
    ConditionalCodeTransformer,
    TransformerConfig,
)
from vqvae.paired_data import NUM_VERTEBRA_CLASSES, PairedXrayDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the SVV-Net conditional code Transformer."
    )
    parser.add_argument("--vqvae-checkpoint", required=True)
    parser.add_argument("--preoperative-dir", required=True)
    parser.add_argument("--postoperative-dir", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output-dir", default="runs/transformer")
    parser.add_argument("--resume")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--encoder-layers", type=int, default=6)
    parser.add_argument("--decoder-layers", type=int, default=6)
    parser.add_argument("--feedforward-dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--save-every",
        type=int,
        default=3,
        help="Save latest and numbered checkpoints every N epochs.",
    )
    parser.add_argument(
        "--preview-every",
        type=int,
        default=3,
        help="Generate a comparison preview every N epochs.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


@torch.no_grad()
def encode_images(vqvae, images: torch.Tensor) -> torch.Tensor:
    _, indices, _ = vqvae.encode(images)
    return indices.flatten(1)


def save_transformer_checkpoint(
    path: Path,
    model: ConditionalCodeTransformer,
    optimizer: AdamW,
    epoch: int,
    latent_shape,
    best_loss: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": model.config.to_dict(),
            "latent_shape": list(latent_shape),
            "best_loss": best_loss,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    if args.height % 16 or args.width % 16:
        raise ValueError("height and width must both be divisible by 16")
    if args.save_every < 1 or args.preview_every < 1:
        raise ValueError("save-every and preview-every must be positive")
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vqvae = load_model(args.vqvae_checkpoint, device)
    vqvae.eval()
    vqvae.requires_grad_(False)

    dataset = PairedXrayDataset(
        args.preoperative_dir,
        args.postoperative_dir,
        args.metadata,
        image_size=(args.height, args.width),
        channels=vqvae.config.image_channels,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    example = dataset[0]["preoperative"].unsqueeze(0).to(device)
    with torch.no_grad():
        _, example_indices, _ = vqvae.encode(example)
    latent_shape = example_indices.shape[1:]
    num_image_tokens = int(example_indices[0].numel())

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        config = TransformerConfig.from_dict(checkpoint["config"])
    else:
        config = TransformerConfig(
            codebook_size=vqvae.config.num_codebook_vectors,
            num_vertebra_classes=NUM_VERTEBRA_CLASSES,
            num_image_tokens=num_image_tokens,
            d_model=args.d_model,
            num_heads=args.num_heads,
            num_encoder_layers=args.encoder_layers,
            num_decoder_layers=args.decoder_layers,
            dim_feedforward=args.feedforward_dim,
            dropout=args.dropout,
        )
    if config.num_image_tokens != num_image_tokens:
        raise ValueError(
            "Transformer checkpoint token count does not match image size: "
            f"{config.num_image_tokens} vs {num_image_tokens}"
        )

    model = ConditionalCodeTransformer(config).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    start_epoch = 0
    best_loss = float("inf")
    if args.resume:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint.get("epoch", 0))
        best_loss = float(checkpoint.get("best_loss", float("inf")))

    for epoch in range(start_epoch + 1, args.epochs + 1):
        model.train()
        progress = tqdm(loader, desc=f"epoch {epoch}/{args.epochs}")
        last_batch = None
        total_epoch_loss = 0.0
        num_batches = 0
        for batch in progress:
            preoperative = batch["preoperative"].to(device)
            postoperative = batch["postoperative"].to(device)
            uiv = batch["uiv"].to(device)
            liv = batch["liv"].to(device)
            with torch.no_grad():
                pre_tokens = encode_images(vqvae, preoperative)
                post_tokens = encode_images(vqvae, postoperative)

            loss = model.training_loss(pre_tokens, post_tokens, uiv, liv)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_epoch_loss += loss.detach().item()
            num_batches += 1

            with torch.no_grad():
                bos = torch.full(
                    (post_tokens.shape[0], 1),
                    model.bos_token,
                    device=device,
                    dtype=torch.long,
                )
                decoder_input = torch.cat(
                    [bos, post_tokens[:, :-1]], dim=1
                )
                predictions = model(
                    pre_tokens, uiv, liv, decoder_input
                ).argmax(dim=-1)
                accuracy = (predictions == post_tokens).float().mean()
            last_batch = (preoperative, postoperative, pre_tokens, uiv, liv)
            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                token_accuracy=f"{accuracy.item():.3f}",
            )
        average_loss = total_epoch_loss / max(num_batches, 1)
        print(f"epoch {epoch} average_loss={average_loss:.6f}")

        if epoch % args.preview_every == 0 and last_batch is not None:
            preoperative, postoperative, pre_tokens, uiv, liv = last_batch
            generated = model.generate(
                pre_tokens[:1], uiv[:1], liv[:1], sample=False
            )
            generated = generated.view(1, *latent_shape)
            predicted_image = vqvae.decode_indices(generated)
            preview = torch.cat(
                [preoperative[:1], postoperative[:1], predicted_image]
            )
            save_image(
                preview.add(1).div(2).clamp(0, 1),
                output_dir / f"preview_{epoch:04d}.png",
                nrow=3,
            )

        if average_loss < best_loss:
            best_loss = average_loss
            save_transformer_checkpoint(
                output_dir / "best.pt",
                model,
                optimizer,
                epoch,
                latent_shape,
                best_loss,
            )

        if epoch % args.save_every == 0 or epoch == args.epochs:
            save_transformer_checkpoint(
                output_dir / "last.pt",
                model,
                optimizer,
                epoch,
                latent_shape,
                best_loss,
            )


if __name__ == "__main__":
    main()

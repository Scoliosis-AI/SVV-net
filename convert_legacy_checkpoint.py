import argparse
from typing import Dict

import torch

from vqvae import ModelConfig, VQVAE


def rename_key(key: str) -> str:
    key = key.replace("codebook.", "quantizer.")
    key = key.replace(".channel_up.", ".skip.")
    key = key.replace(".gn.gn.", ".norm.")
    key = key.replace(".gn.", ".")
    return key


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a checkpoint created by the original vqgan.py."
    )
    parser.add_argument("source")
    parser.add_argument("destination")
    parser.add_argument("--image-channels", type=int, default=1)
    args = parser.parse_args()

    legacy = torch.load(args.source, map_location="cpu")
    legacy_state: Dict[str, torch.Tensor] = legacy.get("state_dict", legacy)
    converted_state = {rename_key(key): value for key, value in legacy_state.items()}

    config = ModelConfig(image_channels=args.image_channels)
    model = VQVAE(config)
    missing, unexpected = model.load_state_dict(converted_state, strict=False)
    allowed_missing = {
        key for key in missing if ".proj_out." in key or key.endswith("num_batches_tracked")
    }
    real_missing = set(missing) - allowed_missing
    if real_missing or unexpected:
        raise RuntimeError(
            f"conversion mismatch; missing={sorted(real_missing)}, "
            f"unexpected={sorted(unexpected)}"
        )

    torch.save(
        {
            "epoch": int(legacy.get("epoch", 0)),
            "model": model.state_dict(),
            "config": config.to_dict(),
        },
        args.destination,
    )
    print(f"converted checkpoint saved to {args.destination}")


if __name__ == "__main__":
    main()

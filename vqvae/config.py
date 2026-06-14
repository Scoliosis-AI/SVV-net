from dataclasses import asdict, dataclass
from typing import Any, Dict, Tuple


@dataclass(frozen=True)
class ModelConfig:
    image_channels: int = 1
    latent_dim: int = 256
    num_codebook_vectors: int = 256
    commitment_cost: float = 0.25
    channels: Tuple[int, ...] = (128, 128, 128, 256, 256, 512)
    decoder_channels: Tuple[int, ...] = (512, 256, 256, 128, 128)
    num_res_blocks: int = 2
    attention_resolutions: Tuple[int, ...] = (16,)
    base_resolution: int = 256

    def to_dict(self) -> Dict[str, Any]:
        config = asdict(self)
        config["channels"] = list(self.channels)
        config["decoder_channels"] = list(self.decoder_channels)
        config["attention_resolutions"] = list(self.attention_resolutions)
        return config

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "ModelConfig":
        values = dict(config)
        if "channels" in values:
            values["channels"] = tuple(values["channels"])
        if "decoder_channels" in values:
            values["decoder_channels"] = tuple(values["decoder_channels"])
        if "attention_resolutions" in values:
            values["attention_resolutions"] = tuple(
                values["attention_resolutions"]
            )
        return cls(**values)

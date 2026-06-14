from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class TransformerConfig:
    codebook_size: int = 256
    num_vertebra_classes: int = 18
    num_image_tokens: int = 256
    d_model: int = 512
    num_heads: int = 8
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    dim_feedforward: int = 1024
    dropout: float = 0.1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "TransformerConfig":
        return cls(**values)


class ConditionalCodeTransformer(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.bos_token = config.codebook_size
        self.code_embedding = nn.Embedding(
            config.codebook_size + 1, config.d_model
        )
        self.uiv_embedding = nn.Embedding(
            config.num_vertebra_classes, config.d_model
        )
        self.liv_embedding = nn.Embedding(
            config.num_vertebra_classes, config.d_model
        )
        self.source_type_embedding = nn.Embedding(3, config.d_model)
        self.source_position = nn.Parameter(
            torch.zeros(1, config.num_image_tokens + 2, config.d_model)
        )
        self.target_position = nn.Parameter(
            torch.zeros(1, config.num_image_tokens, config.d_model)
        )
        self.transformer = nn.Transformer(
            d_model=config.d_model,
            nhead=config.num_heads,
            num_encoder_layers=config.num_encoder_layers,
            num_decoder_layers=config.num_decoder_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
        )
        self.output = nn.Linear(config.d_model, config.codebook_size)
        nn.init.normal_(self.source_position, std=0.02)
        nn.init.normal_(self.target_position, std=0.02)

    def encode_condition(
        self,
        preoperative_tokens: torch.Tensor,
        uiv: torch.Tensor,
        liv: torch.Tensor,
    ) -> torch.Tensor:
        if preoperative_tokens.shape[1] != self.config.num_image_tokens:
            raise ValueError(
                f"expected {self.config.num_image_tokens} image tokens, "
                f"got {preoperative_tokens.shape[1]}"
            )
        batch_size = preoperative_tokens.shape[0]
        device = preoperative_tokens.device
        uiv_token = self.uiv_embedding(uiv).unsqueeze(1)
        liv_token = self.liv_embedding(liv).unsqueeze(1)
        image_tokens = self.code_embedding(preoperative_tokens)
        source = torch.cat([uiv_token, liv_token, image_tokens], dim=1)
        token_types = torch.cat(
            [
                torch.zeros(batch_size, 1, device=device, dtype=torch.long),
                torch.ones(batch_size, 1, device=device, dtype=torch.long),
                torch.full(
                    (batch_size, self.config.num_image_tokens),
                    2,
                    device=device,
                    dtype=torch.long,
                ),
            ],
            dim=1,
        )
        return (
            source
            + self.source_type_embedding(token_types)
            + self.source_position
        )

    def forward(
        self,
        preoperative_tokens: torch.Tensor,
        uiv: torch.Tensor,
        liv: torch.Tensor,
        decoder_tokens: torch.Tensor,
    ) -> torch.Tensor:
        source = self.encode_condition(preoperative_tokens, uiv, liv)
        target_length = decoder_tokens.shape[1]
        target = (
            self.code_embedding(decoder_tokens)
            + self.target_position[:, :target_length]
        )
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            target_length
        ).to(decoder_tokens.device)
        hidden = self.transformer(
            source,
            target,
            tgt_mask=causal_mask,
        )
        return self.output(hidden)

    def training_loss(
        self,
        preoperative_tokens: torch.Tensor,
        postoperative_tokens: torch.Tensor,
        uiv: torch.Tensor,
        liv: torch.Tensor,
    ) -> torch.Tensor:
        bos = torch.full(
            (postoperative_tokens.shape[0], 1),
            self.bos_token,
            device=postoperative_tokens.device,
            dtype=torch.long,
        )
        decoder_tokens = torch.cat(
            [bos, postoperative_tokens[:, :-1]], dim=1
        )
        logits = self(preoperative_tokens, uiv, liv, decoder_tokens)
        return F.cross_entropy(
            logits.reshape(-1, self.config.codebook_size),
            postoperative_tokens.reshape(-1),
        )

    @torch.no_grad()
    def generate(
        self,
        preoperative_tokens: torch.Tensor,
        uiv: torch.Tensor,
        liv: torch.Tensor,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        sample: bool = False,
    ) -> torch.Tensor:
        was_training = self.training
        self.eval()
        generated = torch.full(
            (preoperative_tokens.shape[0], 1),
            self.bos_token,
            device=preoperative_tokens.device,
            dtype=torch.long,
        )
        for _ in range(self.config.num_image_tokens):
            logits = self(
                preoperative_tokens,
                uiv,
                liv,
                generated,
            )[:, -1]
            logits = logits / max(temperature, 1e-6)
            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.shape[-1]))
                logits[logits < values[:, [-1]]] = -torch.inf
            if sample:
                next_token = torch.multinomial(
                    torch.softmax(logits, dim=-1), 1
                )
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
        self.train(was_training)
        return generated[:, 1:]

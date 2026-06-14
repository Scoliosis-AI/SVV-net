from typing import Tuple

import torch
from torch import nn


class VectorQuantizer(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(
            -1.0 / num_embeddings,
            1.0 / num_embeddings,
        )

    def forward(
        self, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_hwc = z.permute(0, 2, 3, 1).contiguous()
        flat_z = z_hwc.reshape(-1, self.embedding_dim)

        distances = (
            flat_z.square().sum(dim=1, keepdim=True)
            + self.embedding.weight.square().sum(dim=1)
            - 2 * flat_z @ self.embedding.weight.t()
        )
        indices = distances.argmin(dim=1)
        quantized = self.embedding(indices).view_as(z_hwc)

        codebook_loss = (quantized - z_hwc.detach()).square().mean()
        commitment_loss = (quantized.detach() - z_hwc).square().mean()
        loss = codebook_loss + self.commitment_cost * commitment_loss

        quantized = z_hwc + (quantized - z_hwc).detach()
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        index_map = indices.view(z.shape[0], z.shape[2], z.shape[3])
        return quantized, index_map, loss

    def lookup(self, indices: torch.Tensor) -> torch.Tensor:
        quantized = self.embedding(indices)
        return quantized.permute(0, 3, 1, 2).contiguous()

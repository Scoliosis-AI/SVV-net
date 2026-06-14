import torch

from vqvae import ModelConfig, VQVAE
from vqvae.conditional_transformer import (
    ConditionalCodeTransformer,
    TransformerConfig,
)
from vqvae.paired_data import encode_vertebra


def test_forward_and_backward():
    config = ModelConfig(
        latent_dim=32,
        num_codebook_vectors=16,
        channels=(32, 32, 32),
        decoder_channels=(32, 32),
        attention_resolutions=(),
        base_resolution=32,
        num_res_blocks=1,
    )
    model = VQVAE(config)
    images = torch.randn(1, 1, 32, 32)
    reconstructed, indices, quantization_loss = model(images)

    assert reconstructed.shape == images.shape
    assert indices.shape == (1, 16, 16)
    (reconstructed.abs().mean() + quantization_loss).backward()


def test_narrow_image_shape():
    config = ModelConfig(
        latent_dim=32,
        num_codebook_vectors=16,
        channels=(32, 32, 32, 32, 32, 32),
        decoder_channels=(32, 32, 32, 32, 32),
        attention_resolutions=(),
        base_resolution=256,
        num_res_blocks=1,
    )
    model = VQVAE(config)
    images = torch.randn(1, 1, 64, 16)

    with torch.no_grad():
        reconstructed, indices, _ = model(images)

    assert reconstructed.shape == images.shape
    assert indices.shape == (1, 4, 1)


def test_high_resolution_architecture_downsamples_five_times():
    config = ModelConfig(
        latent_dim=32,
        num_codebook_vectors=16,
        channels=(32, 32, 32, 32, 32, 32, 32),
        decoder_channels=(32, 32, 32, 32, 32, 32),
        attention_resolutions=(),
        base_resolution=1024,
        num_res_blocks=1,
    )
    model = VQVAE(config)
    images = torch.randn(1, 1, 64, 32)

    with torch.no_grad():
        reconstructed, indices, _ = model(images)

    assert reconstructed.shape == images.shape
    assert indices.shape == (1, 2, 1)


def test_conditional_transformer_loss_and_generation():
    config = TransformerConfig(
        codebook_size=16,
        num_image_tokens=4,
        d_model=32,
        num_heads=4,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=64,
        dropout=0.0,
    )
    model = ConditionalCodeTransformer(config)
    preoperative = torch.randint(0, 16, (2, 4))
    postoperative = torch.randint(0, 16, (2, 4))
    uiv = torch.tensor([1, 2])
    liv = torch.tensor([13, 17])

    loss = model.training_loss(preoperative, postoperative, uiv, liv)
    loss.backward()
    generated = model.generate(preoperative, uiv, liv)

    assert loss.ndim == 0
    assert generated.shape == (2, 4)
    assert generated.min() >= 0
    assert generated.max() < 16


def test_vertebra_encoding():
    assert encode_vertebra("T1") == 1
    assert encode_vertebra("T12") == 12
    assert encode_vertebra("L1") == 13
    assert encode_vertebra("L5") == 17

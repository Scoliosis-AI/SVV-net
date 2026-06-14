# SVV-Net

**SVV-Net** is a conditional generative network for postoperative spinal
X-ray synthesis. It predicts postoperative radiographs from preoperative
images while explicitly conditioning on the upper instrumented vertebra
(UIV) and lower instrumented vertebra (LIV).

The project follows a three-stage pipeline:

1. Train a shared VQ-VAE codebook on preoperative and postoperative X-rays.
2. Freeze the VQ-VAE and train a conditional Transformer to predict
   postoperative image tokens from preoperative tokens, UIV, and LIV.
3. Decode the predicted tokens into a postoperative X-ray.

The default configuration processes single-channel images with a height of
1024 pixels and a width of 256 pixels. The high-resolution VQ-VAE downsamples
the image five times and produces a `32 x 8` latent map containing 256 visual
tokens.

> This repository contains source code only. Do not commit patient images,
> metadata files, model checkpoints, or generated results.

## Features

- Conditional postoperative spinal X-ray generation
- Explicit UIV and LIV conditioning
- High-resolution `1024 x 256` grayscale image support
- Discrete visual representation using VQ-VAE
- Autoregressive conditional Transformer
- Optional PatchGAN training
- CUDA automatic mixed precision
- Checkpoint recovery and preview generation

## Architecture

```text
Preoperative X-ray
        |
        v
  VQ-VAE Encoder ----> Preoperative visual tokens
                              |
UIV embedding ----------------|
LIV embedding ----------------|--> Conditional Transformer
                                      |
                                      v
                           Postoperative visual tokens
                                      |
                                      v
                              VQ-VAE Decoder
                                      |
                                      v
                           Generated postoperative X-ray
```

## Repository Structure

```text
vqvae/
  checkpoint.py
  conditional_transformer.py
  config.py
  data.py
  discriminator.py
  model.py
  modules.py
  paired_data.py
  quantizer.py
tests/
  test_smoke.py
train.py
train_transformer.py
reconstruct.py
generate_postoperative.py
convert_legacy_checkpoint.py
requirements.txt
```

## Requirements

Recommended environment:

- Python 3.10 or 3.11
- NVIDIA GPU with CUDA support
- PyTorch 2.3 or newer

Install the dependencies:

```cmd
python -m pip install -r requirements.txt
```

Check CUDA availability:

```cmd
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Run the tests:

```cmd
python -m pytest -q
```

## Private Dataset Layout

The dataset is not included in this repository. Organize private data locally:

```text
data/
  train/
    pre_op/
      CASE001.jpg
      CASE002.jpg
    post_op/
      CASE001.jpg
      CASE002.jpg
  test/
    pre_op/
      CASE101.jpg
    post_op/
      CASE101.jpg
  metadata.xlsx
```

Each preoperative image must have a postoperative image with the same file
name. The metadata file must contain the following columns:

| case_id | UIV | LIV |
|---|---|---|
| CASE001 | T4 | L2 |
| CASE002 | T3 | L1 |
| CASE101 | T5 | L3 |

`case_id` must match the image file name without its extension. Excel `.xlsx`
and `.xls` files are supported. Store `case_id` as text, especially when IDs
contain leading zeros.

The training and test case IDs must not overlap. Dataset overlap causes data
leakage and invalidates generalization results.

## Stage 1: Train the VQ-VAE

Windows CMD:

```cmd
python -u train.py ^
  --dataset data\train\pre_op data\train\post_op ^
  --output-dir runs\vqvae_highres ^
  --model-preset highres ^
  --height 1024 ^
  --width 256 ^
  --batch-size 4 ^
  --epochs 500 ^
  --num-workers 4 ^
  --augment ^
  --amp ^
  --channels-last ^
  --device cuda
```

The training outputs are written to:

```text
runs/vqvae_highres/best.pt
runs/vqvae_highres/last.pt
runs/vqvae_highres/reconstruction_XXXX.png
```

Resume interrupted VQ-VAE training:

```cmd
python -u train.py ^
  --dataset data\train\pre_op data\train\post_op ^
  --output-dir runs\vqvae_highres ^
  --height 1024 ^
  --width 256 ^
  --batch-size 4 ^
  --epochs 500 ^
  --num-workers 4 ^
  --augment ^
  --amp ^
  --channels-last ^
  --device cuda ^
  --resume runs\vqvae_highres\last.pt
```

Reduce the batch size to `2` or `1` if CUDA runs out of memory.

## Inspect VQ-VAE Reconstructions

```cmd
python -u reconstruct.py ^
  --checkpoint runs\vqvae_highres\best.pt ^
  --input-dir data\test\pre_op ^
  --output-dir reconstructions ^
  --height 1024 ^
  --width 256 ^
  --batch-size 4 ^
  --device cuda
```

## Stage 2: Train the Conditional Transformer

```cmd
python -u train_transformer.py ^
  --vqvae-checkpoint runs\vqvae_highres\best.pt ^
  --preoperative-dir data\train\pre_op ^
  --postoperative-dir data\train\post_op ^
  --metadata data\metadata.xlsx ^
  --output-dir runs\transformer ^
  --height 1024 ^
  --width 256 ^
  --batch-size 4 ^
  --epochs 500 ^
  --num-workers 4 ^
  --device cuda
```

The Transformer receives:

```text
preoperative image tokens + UIV embedding + LIV embedding
```

Its training target is the corresponding sequence of postoperative image
tokens.

Resume interrupted Transformer training:

```cmd
python -u train_transformer.py ^
  --vqvae-checkpoint runs\vqvae_highres\best.pt ^
  --preoperative-dir data\train\pre_op ^
  --postoperative-dir data\train\post_op ^
  --metadata data\metadata.xlsx ^
  --output-dir runs\transformer ^
  --height 1024 ^
  --width 256 ^
  --batch-size 4 ^
  --epochs 500 ^
  --num-workers 4 ^
  --device cuda ^
  --resume runs\transformer\last.pt
```

## Stage 3: Generate Postoperative X-rays

```cmd
python -u generate_postoperative.py ^
  --vqvae-checkpoint runs\vqvae_highres\best.pt ^
  --transformer-checkpoint runs\transformer\best.pt ^
  --preoperative-dir data\test\pre_op ^
  --metadata data\metadata.xlsx ^
  --output-dir generated_postoperative ^
  --height 1024 ^
  --width 256 ^
  --device cuda
```

Generation uses deterministic greedy decoding by default. To enable
stochastic sampling, add:

```cmd
--sample --temperature 1.0 --top-k 20
```

## Important Notes

- `1024 x 256` means `height=1024` and `width=256`.
- All three stages must use the same image dimensions.
- The `highres` architecture is not compatible with older `512 x 128`
  checkpoints and must be trained from scratch.
- Use `best.pt` for final inference and `last.pt` for training recovery.
- Generated images are intended for research only.
- SVV-Net must not be used directly for clinical diagnosis or treatment
  decisions.

## License

This project is distributed under the terms of the included MIT License.

# Conditional Postoperative X-ray Generation

这是一个三阶段的条件式术后 X 光片生成项目：

1. 使用术前和术后 X 光片训练共享的 VQ-VAE codebook。
2. 冻结 VQ-VAE，以术前图像 token、UIV 和 LIV 为输入训练 Transformer。
3. 使用术前图像、UIV 和 LIV 生成对应的术后图像。

本版本默认处理单通道、**高 1024 × 宽 256** 的图像。高分辨率 VQ-VAE
经过 5 次下采样，得到 `32 × 8 = 256` 个离散 token，避免高分辨率直接产生
1024 个 token 而显著增加 Transformer 的显存和推理时间。

> 本仓库只包含代码。请勿上传患者图像、Excel 信息表、模型权重或生成结果。

## 项目结构

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

## 环境

推荐：

- Python 3.10 或 3.11
- 支持 CUDA 的 NVIDIA GPU
- PyTorch 2.3 或更新版本

安装依赖：

```cmd
python -m pip install -r requirements.txt
```

验证 CUDA：

```cmd
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

运行测试：

```cmd
python -m pytest -q
```

## 私有数据结构

数据不应提交到 GitHub。建议在本地按以下方式组织：

```text
data/
  train/
    pre_op/
      001.jpg
    post_op/
      001.jpg
  test/
    pre_op/
      101.jpg
    post_op/
      101.jpg
  metadata.xlsx
```

术前和术后图像必须使用相同文件名。信息表至少包含：

| 登记片号 | UIV | LIV |
|---|---|---|
| 001 | T4 | L2 |
| 101 | T3 | L1 |

`登记片号` 必须与图片文件名（不含扩展名）一致。支持 `.xlsx` 和 `.xls`。

训练集和测试集的病例 ID 必须完全不重叠，否则会造成数据泄漏，测试指标不能
代表模型对新病例的泛化效果。

## 第一阶段：VQ-VAE

Windows CMD：

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

输出：

```text
runs/vqvae_highres/best.pt
runs/vqvae_highres/last.pt
runs/vqvae_highres/reconstruction_XXXX.png
```

恢复训练：

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

显存不足时将 `--batch-size 4` 改为 `2` 或 `1`。

## VQ-VAE 重建检查

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

## 第二阶段：条件 Transformer

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

Transformer 的条件输入为：

```text
术前图像 token + UIV embedding + LIV embedding
```

训练目标为对应的术后图像 token。

恢复训练：

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

## 第三阶段：生成术后图像

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

默认使用确定性的 greedy decoding。随机采样可增加：

```cmd
--sample --temperature 1.0 --top-k 20
```

## 注意事项

- `1024 × 256` 在代码中表示 `height=1024, width=256`。
- 第一、二、三阶段必须使用相同的图像尺寸。
- `highres` 结构与旧版 `512 × 128` 权重不兼容，需要重新训练。
- `best.pt` 用于最终推理，`last.pt` 用于中断后恢复训练。
- 生成结果仅用于科研验证，不能直接用于临床诊断或治疗决策。

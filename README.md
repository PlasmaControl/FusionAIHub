# Fusion AI Toolkit & Hub (FAITH)

## Enviornment Setup
1. Install Pixi
```bash
curl -fsSL https://pixi.sh/install.sh | sh
```
2. Create a new environment
```bash
pixi install
```
3. Activate the environment
```bash
pixi shell
```

## Datas
Unprocessed data is stored in `/scratch/gpfs/EKOLEMEN/d3d_fusion_data/`.

Unprocessed videos are temporarily stored in `/scratch/gpfs/EKOLEMEN/big_d3d_data/images/`.

Model-ready files are stored in `/scratch/gpfs/EKOLEMEN/foundation_model/`.

Model-ready files should be set with `664` permissions at least.

## Flash Attention
10x speedup, 10x memory reduction.

Flash Attention is a fast attention mechanism that can be used to speed up the attention mechanism in the model.
Flash-Attention-2 reduces memory usage by 10-20x and increases speed by 9x compared to standard attention.
Installation depends on the CUDA, Python, and PyTorch versions.

DO NOT USE `pip install flash-attn` since building wheels will take a long time.
Instead, search for a matching wheel for your system from either of the following sources:
- https://github.com/Dao-AILab/flash-attention/releases
- https://github.com/mjun0812/flash-attention-prebuild-wheels/releases

Make sure your GCC version is at least 9 or higher. On Princeton clusters, you should upgrade it via
```bash
module load gcc-toolset/10
```
Then, install flash-attn via
```bash
wget <url>
pip install ninja
pip install <wheel>
```
A pre-downloaded wheel will be made available soon. For now, the link for the wheel on Princeton clusters is:
https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.6.3+cu128torch2.10-cp311-cp311-linux_x86_64.whl
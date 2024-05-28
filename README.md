# Simple DiLoCo

This repo contains a minimal reproducible torch example of the ["DiLoCo: Distributed Low-Communication Training of Language Models"](https://arxiv.org/abs/2311.08105) approach in 180 lines of code.

## How to run the code

First install the dependencies :

```bash
pip install -r requirements.txt
```

## Start run

### 1 DiLoCo replica worker

```bash
torchrun --nproc_per_node=1  pure_torch_diloco.py --per-device-train-batch-size 16 --batch-size 256 --lr 1e-3 --warmup-steps 50  --local-steps 10
```

### 2 DiLoCo replica workers

```bash
torchrun --nproc_per_node=2  pure_torch_diloco.py --per-device-train-batch-size 16 --batch-size 256 --lr 1e-3 --warmup-steps 50  --local-steps 10
```
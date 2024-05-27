# Simple diloco

how to run the code :


first install the dependencies :

```bash
pip install -r requirements.txt
```

## start run

### 1 diloco replica

```bash
torchrun --nproc_per_node=1  pure_torch_diloco.py --per-device-train-batch-size 16 --batch-size 256 --lr 1e-3 --warmup-steps 50  --local-steps 10
```

### 2 diloco replica

```bash
torchrun --nproc_per_node=2  pure_torch_diloco.py --per-device-train-batch-size 16 --batch-size 256 --lr 1e-3 --warmup-steps 50  --local-steps 10
```
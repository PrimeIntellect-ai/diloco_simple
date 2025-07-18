# Simple DiLoCo

This repo contains a minimal reproducible torch example of the ["DiLoCo: Distributed Low-Communication Training of Language Models"](https://arxiv.org/abs/2311.08105) approach in 180 lines of code.

## Prerequisites

The script uses the Mistral-7B tokenizer, which is now a gated model on Hugging Face. You need to:

1. Create a Hugging Face account at https://huggingface.co/
2. Request access to the model at https://huggingface.co/mistralai/Mistral-7B-v0.1
3. Generate an access token at https://huggingface.co/settings/tokens
4. Set your Hugging Face token:
   ```bash
   export HF_TOKEN=hf_YOUR_TOKEN_HERE
   ```
   
   Alternatively, you can login using the Hugging Face CLI:
   ```bash
   huggingface-cli login
   # Enter your access token when prompted
   ```

## How to run the code

First install the dependencies :

```bash
pip install -r requirements.txt
```

### Using uv (recommended for isolated environment)

To create a completely isolated environment using uv with a specific Python version:

```bash
# Install uv if not already installed
pip install uv

# Create a virtual environment with Python 3.12
uv venv venv-py312 --python 3.12

# Activate the virtual environment
source venv-py312/bin/activate

# Install all requirements
uv pip install -r requirements.txt
```

## Start run

By default, the script will try to log to Weights & Biases. To run in offline mode:

```bash
export WANDB_MODE=offline
```

If you want to use Weights & Biases logging, first login:
```bash
wandb login
```

### 1 DiLoCo replica worker

```bash
torchrun --nproc_per_node=1  pure_torch_diloco.py --per-device-train-batch-size 16 --batch-size 256 --lr 1e-3 --warmup-steps 50  --local-steps 10
```

### 2 DiLoCo replica workers

```bash
torchrun --nproc_per_node=2  pure_torch_diloco.py --per-device-train-batch-size 16 --batch-size 256 --lr 1e-3 --warmup-steps 50  --local-steps 10
```
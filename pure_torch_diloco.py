import os

import torch
import torch.distributed as dist
import wandb
from cyclopts import App
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
from torch.distributed import destroy_process_group, init_process_group
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    LlamaConfig,
    LlamaForCausalLM,
    get_cosine_schedule_with_warmup,
)


# Function to initialize the distributed process group
def ddp_setup():
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def get_offloaded_param(outer_optimizer: torch.optim.Optimizer):
    return [
        param.data.detach().clone().to("cpu")
        for group in outer_optimizer.param_groups
        for param in group["params"]
    ]


app = App()


@app.default
def main(
    batch_size: int = 512,
    per_device_train_batch_size: int = 32,
    seq_length: int = 1024,
    warmup_steps: int = 1000,
    total_steps: int = 88_000,
    project: str = "diloco",
    config_path: str = "config_14m.json",
    lr: float = 4e-4,
    outer_lr: float = 0.7,
    local_steps: int = 500,
):
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    assert batch_size % per_device_train_batch_size == 0
    gradient_accumulation_steps = batch_size / per_device_train_batch_size

    if local_rank == 0:
        wandb.init(project=project)

    # Load model configuration and tokenizer
    config = LlamaConfig.from_pretrained(pretrained_model_name_or_path=config_path)

    model = LlamaForCausalLM(config).to(local_rank)

    for param in model.parameters():
        # this make sure all device have the same weight init
        dist.broadcast(param.data, src=0)

    # Setup optimizers
    inner_optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95))
    outer_optimizer = torch.optim.SGD(
        model.parameters(), lr=outer_lr, momentum=0.9, nesterov=True
    )

    params_offloaded = get_offloaded_param(outer_optimizer)

    scheduler = get_cosine_schedule_with_warmup(
        inner_optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        "mistralai/Mistral-7B-v0.1", use_fast=True
    )
    tokenizer.pad_token = "</s>"  # Ensure pad token is set for models that need it

    ds = load_dataset("PrimeIntellect/c4-tiny", "en", ignore_verifications=True)

    def tokenize_function(data):
        outputs = tokenizer(data["text"], truncation=True, max_length=seq_length)
        return outputs

    tokenized_datasets = ds.map(
        tokenize_function, batched=True, remove_columns=["text", "timestamp", "url"]
    )
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    train_dataset = split_dataset_by_node(
        tokenized_datasets["train"], world_size=world_size, rank=local_rank
    )
    train_dataloader = DataLoader(
        train_dataset, collate_fn=data_collator, batch_size=per_device_train_batch_size
    )

    model.train()

    loss_batch = 0

    for step, batch in enumerate(iterable=train_dataloader):

        real_step = (step + 1) // gradient_accumulation_steps
        step_within_grad_acc = (step + 1) % gradient_accumulation_steps

        for key in batch.keys():
            batch[key] = batch[key].to("cuda")

        outputs = model(**batch)
        loss = outputs.loss / gradient_accumulation_steps

        loss_batch += loss.detach()

        loss.backward()

        if step_within_grad_acc == 0:

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # gradient clipping

            inner_optimizer.step()
            scheduler.step()

            inner_optimizer.zero_grad()

            if real_step % local_steps == 0:
                if local_rank == 0:
                    print(f"perform outer step at step {real_step}")

                main_param = [
                    param
                    for group in inner_optimizer.param_groups
                    for param in group["params"]
                ]

                for param_offloaded, param in zip(params_offloaded, main_param):
                    param.grad = param_offloaded.data.to(param.device) - param.data

                for param in [
                    param
                    for group in outer_optimizer.param_groups
                    for param in group["params"]
                ]:
                    if param.requires_grad and param.grad is not None:
                        dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)

                outer_optimizer.step()
                outer_optimizer.zero_grad()
                params_offloaded = get_offloaded_param(outer_optimizer)

            if local_rank == 0:

                dict_to_log = {
                    "Loss": loss_batch.item(),
                    "step": real_step,
                    "lr": [group["lr"] for group in inner_optimizer.param_groups][0],
                    "Perplexity": torch.exp(loss_batch).item(),
                    "effective_step": real_step * world_size,
                    "total_samples": real_step * batch_size * world_size,
                }

                wandb.log(dict_to_log)
                print(
                    f"step: {real_step}, loss: {loss_batch.item()}, lr {[group['lr'] for group in inner_optimizer.param_groups][0]}"
                )
                loss_batch = 0

    print("Training completed.")
    wandb.finish()


if __name__ == "__main__":
    ddp_setup()
    app()
    destroy_process_group()

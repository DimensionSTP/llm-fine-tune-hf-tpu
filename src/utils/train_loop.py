import math

from omegaconf import DictConfig

import torch
from torch.utils.data import DataLoader
from torch import optim

from torch_xla.core import xla_model as xm
from torch_xla.distributed import fsdp
from torch_xla.distributed import parallel_loader
from torch_xla.distributed.fsdp import XlaFullyShardedDataParallel as FSDP
from torch_xla.distributed.fsdp.wrap import wrap, ModuleWrapPolicy

from transformers import AutoModelForCausalLM

import wandb


def train_loop(
    config: DictConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    model: AutoModelForCausalLM,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LambdaLR,
) -> None:
    wandb.init(
        project=f"{config.model_type}-{config.mode}",
        name=f"{config.model_type}-max_length={config.max_length}-lr={config.lr}",
    )

    device = xm.xla_device()
    print(f"Using XLA device: {device}")

    train_loader = parallel_loader.MpDeviceLoader(
        train_loader,
        device,
    )
    val_loader = parallel_loader.MpDeviceLoader(
        val_loader,
        device,
    )

    print("Checking model parameters before wrapping with FSDP:")
    for name, param in model.named_parameters():
        if param.numel() == 0:
            print(f"Warning: Parameter {name} has size 0!")
        print(f"Parameter {name} - shape: {param.shape}")

    xm.rendezvous("before_fsdp")
    try:
        fsdp_model = FSDP(
            model,
            policy=ModuleWrapPolicy({torch.nn.Embedding}),
        ).to(device)
        print("Model successfully wrapped with FSDP.")
    except Exception as e:
        print(f"Error in FSDP wrapping: {e}")
        return

    gradient_accumulation_steps = config.gradient_accumulation_steps

    num_train_samples = len(train_loader._loader.dataset)
    effective_batch_size = (
        config.per_device_train_batch_size
        * config.gradient_accumulation_steps
        * config.tpu_num_cores
    )
    total_steps = (
        math.ceil(num_train_samples / effective_batch_size) * config.num_train_epochs
    )
    print(f"Total steps: {total_steps}, Effective batch size: {effective_batch_size}")

    global_step = 0
    accumulation_loss = 0.0
    for epoch in range(config.num_train_epochs):
        fsdp_model.train()
        print(f"Starting epoch {epoch + 1}/{config.num_train_epochs}")

        for step, inputs in enumerate(train_loader):
            if "input_ids" not in inputs or inputs["input_ids"].size(0) == 0:
                if step == 0:
                    print(f"Warning: Empty batch or missing 'input_ids'")
                continue

            try:
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = fsdp_model(**inputs)
                loss = outputs.loss / gradient_accumulation_steps
                loss.backward()

                accumulation_loss += loss.item()
            except Exception as e:
                if step == 0:
                    print(f"Error during forward/backward pass")
                continue

            if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(
                train_loader
            ):
                try:
                    torch.nn.utils.clip_grad_norm_(
                        fsdp_model.parameters(),
                        config.max_grad_norm,
                    )

                    xm.optimizer_step(optimizer)
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    if global_step % config.logging_steps == 0:
                        wandb.log(
                            {
                                "train_loss": accumulation_loss,
                                "global_step": global_step,
                            }
                        )
                        print(
                            f"Global step {global_step}, Accumulated Loss: {accumulation_loss}"
                        )
                        accumulation_loss = 0.0

                    if global_step % config.save_steps == 0:
                        xm.rendezvous("saving_checkpoint")
                        xm.save(
                            fsdp_model.state_dict(),
                            f"{config.put_dif}/checkpoint-{global_step}.pt",
                        )
                        print(
                            f"checkpoint-{global_step}.pt saved at {config.output_dir}"
                        )

                except Exception as e:
                    print(f"Error during optimizer step at step {global_step}: {e}")
                    continue

        fsdp_model.eval()
        val_loss = 0.0
        for inputs in val_loader:
            if "input_ids" not in inputs or inputs["input_ids"].size(0) == 0:
                print("Warning: Empty batch or missing 'input_ids' during validation")
                continue

            try:
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = fsdp_model(**inputs)
                    val_loss += outputs.loss.item()
            except Exception as e:
                print(f"Error during validation step: {e}")
                continue

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch + 1} Validation Loss: {avg_val_loss}")
        wandb.log({"val_loss": avg_val_loss, "epoch": epoch + 1})

    try:
        xm.rendezvous("saving_model")
        xm.save(fsdp_model.state_dict(), f"{config.output_dir}/final_model.pt")
        print(f"final_model.pt saved at {config.output_dir}")
    except Exception as e:
        print(f"Error saving final model: {e}")

    wandb.finish()

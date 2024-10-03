import math

from omegaconf import DictConfig

import torch
from torch.utils.data import DataLoader
from torch import optim

from torch_xla.core import xla_model as xm
from torch_xla.distributed import fsdp
from torch_xla.distributed import parallel_loader
from torch_xla.distributed.fsdp import XlaFullyShardedDataParallel as FSDP

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
    train_loader = parallel_loader.MpDeviceLoader(
        train_loader,
        device,
    )
    val_loader = parallel_loader.MpDeviceLoader(
        val_loader,
        device,
    )

    fsdp_model = FSDP(model).to(device)

    gradient_accumulation_steps = config.gradient_accumulation_steps

    global_step = 0
    for epoch in range(config.num_train_epochs):
        fsdp_model.train()

        for step, inputs in enumerate(train_loader):
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = fsdp_model(**inputs)
            loss = outputs.loss / gradient_accumulation_steps
            loss.backward()

            if (step + 1) % gradient_accumulation_steps == 0:
                xm.optimizer_step(optimizer)
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                wandb.log({"train_loss": loss.item(), "global_step": global_step})
                print(f"Epoch {epoch + 1}, Step {global_step}, Loss: {loss.item()}")

                if global_step % config.save_steps == 0:
                    xm.rendezvous("saving_checkpoint")
                    xm.save(
                        fsdp_model.state_dict(),
                        f"{config.output_dir}/checkpoint-{global_step}.pt",
                    )
                    print(f"checkpoint-{global_step}.pt saved at {config.output_dir}")

        fsdp_model.eval()
        val_loss = 0.0
        for inputs in val_loader:
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = fsdp_model(**inputs)
                val_loss += outputs.loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch + 1}, Validation Loss: {avg_val_loss}")
        wandb.log({"val_loss": avg_val_loss, "epoch": epoch + 1})

    xm.rendezvous("saving_model")
    xm.save(fsdp_model.state_dict(), f"{config.output_dir}/final_model.pt")
    wandb.finish()

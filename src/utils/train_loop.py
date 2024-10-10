import numpy as np

from omegaconf import DictConfig

import torch
from torch.utils.data import DataLoader
from torch import optim
from torch.distributed import checkpoint as dist_cp

from torch_xla import runtime as xr
from torch_xla.core import xla_model as xm
from torch_xla.distributed import parallel_loader
from torch_xla.experimental.xla_sharding import Mesh
from torch_xla.experimental import xla_sharding as xs
from torch_xla.experimental import distributed_checkpoint as xc

from transformers import AutoModelForCausalLM

import wandb

from .spmd_setup import partition_module


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

    xr.use_spmd()

    device = xm.xla_device()
    print(f"Using XLA device: {device}")

    para_train_loader = parallel_loader.MpDeviceLoader(
        train_loader,
        device,
    )
    para_val_loader = parallel_loader.MpDeviceLoader(
        val_loader,
        device,
    )

    num_devices = xr.global_runtime_device_count()
    mesh_shape = (
        1,
        num_devices,
        1,
        1,
    )
    device_ids = np.array(range(num_devices))
    mesh = Mesh(
        device_ids,
        mesh_shape,
        (
            "dp",
            "fsdp",
            "mp",
            "sp",
        ),
    )

    xm.rendezvous("before_smpd")
    try:
        partition_module(
            model,
            mesh,
            verbose=True,
        )
        print("Model successfully wrapped with SMPD.")
    except Exception as e:
        xm.master_print(f"Error in SMPD wrapping: {e}")
        return

    gradient_accumulation_steps = config.gradient_accumulation_steps

    global_step = 0
    accumulation_loss = 0.0
    for epoch in range(config.num_train_epochs):
        model.train()
        xm.master_print(f"Starting epoch {epoch + 1}/{config.num_train_epochs}")

        for step, batch in enumerate(para_train_loader):
            if "input_ids" not in batch or batch["input_ids"].size(0) == 0:
                if step == 0:
                    xm.master_print(f"Warning: Empty batch or missing 'input_ids'")
                continue

            try:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                xs.mark_sharding(input_ids, mesh, (0, 1))
                xs.mark_sharding(attention_mask, mesh, (0, 1))
                xs.mark_sharding(labels, mesh, (0, 1))
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )

                loss = outputs.loss / gradient_accumulation_steps
                loss.backward()

                accumulation_loss += loss.item()
            except Exception as e:
                if step == 0:
                    xm.master_print(f"Error during forward/backward pass")
                continue

            if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(
                train_loader
            ):
                try:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
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
                        xm.master_print(
                            f"Global step {global_step}, Accumulated Loss: {accumulation_loss}"
                        )
                        accumulation_loss = 0.0

                    if global_step % config.save_steps == 0:
                        xm.rendezvous(f"saving checkpoint-{global_step}")
                        state_dict = {
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                        }
                        dist_cp.save(
                            state_dict=state_dict,
                            storage_writer=dist_cp.FileSystemWriter(
                                f"{config.output_dir}/checkpoint-{global_step}"
                            ),
                            planner=xc.SPMDSavePlanner(),
                        )
                        xm.master_print(
                            f"checkpoint-{global_step} saved at {config.output_dir}"
                        )

                except Exception as e:
                    xm.rendezvous(
                        f"Error during optimizer step at step {global_step}: {e}"
                    )
                    continue

        model.eval()
        val_loss = 0.0
        for inputs in para_val_loader:
            if "input_ids" not in inputs or inputs["input_ids"].size(0) == 0:
                xm.master_print(
                    "Warning: Empty batch or missing 'input_ids' during validation"
                )
                continue

            try:
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = model(**inputs)
                    val_loss += outputs.loss.item()
            except Exception as e:
                xm.master_print(f"Error during validation step: {e}")
                continue

        avg_val_loss = val_loss / len(val_loader)
        xm.master_print(f"Epoch {epoch + 1} Validation Loss: {avg_val_loss}")
        wandb.log(
            {
                "val_loss": avg_val_loss,
                "epoch": epoch + 1,
            }
        )

    try:
        xm.rendezvous(f"saving last checkpoint")
        model_state = model.state_dict()
        xm.all_reduce(
            "sum",
            model_state,
        )
        if xm.is_master_ordinal():
            cpu_model = model.cpu()
            cpu_model.load_state_dict(model_state)
            cpu_model.save_pretrained(f"{config.output_dir}/last")
        xm.master_print(f"last checkpoin saved at {config.output_dir}")
    except Exception as e:
        xm.master_print(f"Error saving final model: {e}")

    wandb.finish()

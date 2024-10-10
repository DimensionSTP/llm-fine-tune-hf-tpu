import os
import math

from omegaconf import DictConfig

import torch
from torch import optim
from torch.utils.data import DataLoader

from torch_xla.distributed import xla_multiprocessing as xmp

from transformers import AutoTokenizer, AutoModelForCausalLM

from ..utils.setup import SetUp
from ..utils.train_loop import train_loop


def train(
    config: DictConfig,
) -> None:
    os.environ["WANDB_PROJECT"] = f"{config.model_type}-{config.mode}"
    os.environ["WANDB_NAME"] = (
        f"{config.model_type}-max_length=${config.max_length}-lr{config.lr}"
    )

    data_encoder = AutoTokenizer.from_pretrained(
        config.model_path,
        use_fast=True,
    )
    if data_encoder.pad_token_id is None:
        data_encoder.pad_token_id = data_encoder.eos_token_id
    data_encoder.padding_side = config.padding_side

    model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        output_hidden_states=False,
        torch_dtype=torch.bfloat16,
    )

    setup = SetUp(config)

    train_dataset = setup.get_train_dataset()
    val_dataset = setup.get_val_dataset()

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.per_device_train_batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config.per_device_eval_batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=True,
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    num_train_samples = len(train_dataset)
    effective_batch_size = (
        config.per_device_train_batch_size
        * config.gradient_accumulation_steps
        * config.tpu_num_cores
    )
    total_steps = (
        math.ceil(num_train_samples / effective_batch_size) * config.num_train_epochs
    )
    print(f"Total steps: {total_steps}, Effective batch size: {effective_batch_size}")

    custom_scheduler = setup.get_scheduler()
    scheduler = custom_scheduler(
        total_steps=total_steps,
        optimizer=optimizer,
    )

    xmp.spawn(
        train_loop(
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
        )
    )

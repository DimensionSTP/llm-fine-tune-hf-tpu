import os
import math

from omegaconf import DictConfig

import torch
from torch import optim
from torch_xla.distributed.fsdp import XlaFullyShardedDataParallel as FSDP

from transformers import AutoTokenizer, AutoModelForCausalLM

from ..utils.setup import SetUp
from ..trainers.tpu_trainer import FSDPTrainer


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
        torch_dtype=torch.float32,
    )
    fsdp_model = FSDP(model)

    setup = SetUp(config)

    train_dataset = setup.get_train_dataset()
    eval_dataset = setup.get_val_dataset()

    training_arguments = setup.get_training_arguments()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    num_train_samples = len(train_dataset)
    effective_batch_size = (
        training_arguments.per_device_train_batch_size
        * training_arguments.gradient_accumulation_steps
        * training_arguments.tpu_num_cores
    )
    total_steps = (
        math.ceil(num_train_samples / effective_batch_size)
        * training_arguments.num_train_epochs
    )

    custom_scheduler = setup.get_scheduler()
    scheduler = custom_scheduler(
        total_steps=total_steps,
        optimizer=optimizer,
    )

    trainer = FSDPTrainer(
        fsdp_model=fsdp_model,
        optimizer=optimizer,
        scheduler=scheduler,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=data_encoder,
        optimizers=(
            None,
            None,
        ),
    )

    trainer.train()

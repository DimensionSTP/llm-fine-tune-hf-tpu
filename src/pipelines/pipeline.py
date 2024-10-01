import os
import math

from omegaconf import DictConfig

import torch
from torch import optim

from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer

from accelerate import Accelerator

import wandb

from ..utils.setup import SetUp


def train(
    config: DictConfig,
) -> None:
    os.environ["WANDB_PROJECT"] = f"{config.model_name}-{config.mode}"
    os.environ["WANDB_NAME"] = (
        f"{config.model_name}-max_length=${config.max_length}-lr{config.lr}"
    )

    data_encoder = AutoTokenizer.from_pretrained(
        config.model_path,
        use_fast=True,
    )

    if config.bf16 == True:
        precision = torch.bfloat16
    else:
        precision = torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        output_hidden_states=False,
        torch_dtype=precision,
        device_map="auto",
    )

    setup = SetUp(config)

    dataloader = setup.get_dataloader()
    dataset = dataloader()
    train_dataset = dataset["train"]
    eval_dataset = dataset["eval"]

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

    accelerator = Accelerator()

    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=data_encoder,
        optimizers=(
            optimizer,
            scheduler,
        ),
        accelerator=accelerator,
    )

    trainer.train()

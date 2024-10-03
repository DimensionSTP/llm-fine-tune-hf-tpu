from typing import Dict, Union, Any

import torch
from torch import nn
from torch_xla.core import xla_model as xm

from transformers import Trainer


class FSDPTrainer(Trainer):
    def __init__(
        self,
        fsdp_model,
        optimizer,
        scheduler,
        *args,
        **kwargs,
    ):
        super().__init__(
            *args,
            **kwargs,
        )
        self.fsdp_model = fsdp_model
        self.optimizer = optimizer
        self.scheduler = scheduler

    def training_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
    ) -> torch.Tensor:
        self.fsdp_model.train()
        model = self.fsdp_model.to(xm.xla_device())

        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(
                model,
                inputs,
            )
        loss = loss / self.args.gradient_accumulation_steps
        loss.backward()

        if (self.state.global_step + 1) % self.args.gradient_accumulation_steps == 0:
            xm.optimizer_step(self.optimizer)
            self.optimizer.zero_grad()

            if self.scheduler is not None:
                self.scheduler.step()

        return loss.detach()

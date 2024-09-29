from omegaconf import DictConfig
from hydra.utils import instantiate

from transformers import TrainingArguments


class SetUp:
    def __init__(
        self,
        config: DictConfig,
    ) -> None:
        self.config = config

    def get_dataset(self) -> object:
        dataset: object = instantiate(
            self.config.dataset,
        )
        return dataset

    def get_training_arguments(self) -> TrainingArguments:
        training_arguments: TrainingArguments = instantiate(
            self.config.preparation,
        )
        return training_arguments

    def get_scheduler(self) -> object:
        scheduler: object = instantiate(
            self.config.scheduler,
        )
        return scheduler

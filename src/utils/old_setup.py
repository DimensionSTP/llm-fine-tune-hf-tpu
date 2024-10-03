from omegaconf import DictConfig
from hydra.utils import instantiate

from torch.utils.data import Dataset

from transformers import TrainingArguments


class SetUp:
    def __init__(
        self,
        config: DictConfig,
    ) -> None:
        self.config = config

    def get_train_dataset(self) -> Dataset:
        train_dataset: Dataset = instantiate(
            self.config.dataset,
            split=self.config.split.train,
        )
        return train_dataset

    def get_val_dataset(self) -> Dataset:
        val_dataset: Dataset = instantiate(
            self.config.dataset,
            split=self.config.split.val,
        )
        return val_dataset

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

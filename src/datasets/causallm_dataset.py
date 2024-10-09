from typing import Dict, Any, List

import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset

from transformers import AutoTokenizer


class StructuralDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        split: str,
        split_ratio: float,
        seed: int,
        instruction_column_name: str,
        data_column_name: str,
        target_column_name: str,
        max_length: int,
        model_path: str,
        padding_side: str,
    ) -> None:
        self.data_path = data_path
        self.split = split
        self.split_ratio = split_ratio
        self.seed = seed
        self.instruction_column_name = instruction_column_name
        self.data_column_name = data_column_name
        self.target_column_name = target_column_name
        self.max_length = max_length
        self.data_encoder = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=True,
        )
        if self.data_encoder.pad_token_id is None:
            self.data_encoder.pad_token_id = self.data_encoder.eos_token_id
        self.data_encoder.padding_side = padding_side
        dataset = self.get_dataset()
        self.instructions = dataset["instructions"]
        self.datas = dataset["datas"]
        self.labels = dataset["labels"]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(
        self,
        idx: int,
    ) -> Dict[str, Any]:
        prompt = self.generate_prompt(
            instruction=self.instructions[idx],
            data=self.datas[idx],
            label=self.labels[idx],
        )
        encoded = self.encode_text(
            data=prompt,
        )
        if "token_type_ids" in encoded.keys():
            del encoded["token_type_ids"]
        encoded["labels"] = encoded["input_ids"]
        return encoded

    def get_dataset(self) -> Dict[str, List[Any]]:
        if self.split in ["train", "val"]:
            parquet_path = f"{self.data_path}/train.parquet"
            data = pd.read_parquet(parquet_path)
            data = data.fillna("_")
            train_data, val_data = train_test_split(
                data,
                test_size=self.split_ratio,
                random_state=self.seed,
                shuffle=True,
            )
            if self.split == "train":
                data = train_data
            else:
                data = val_data
        else:
            raise ValueError(f"Inavalid split: {self.split}")
        instructions = (
            data[self.instruction_column_name].apply(lambda x: x.strip()).tolist()
        )
        datas = data[self.data_column_name].apply(lambda x: x.strip()).tolist()
        labels = data[self.target_column_name].apply(lambda x: x.strip()).tolist()
        return {
            "instructions": instructions,
            "datas": datas,
            "labels": labels,
        }

    def encode_text(
        self,
        data: str,
    ) -> Dict[str, torch.Tensor]:
        encoded = self.data_encoder(
            data,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=True,
        )
        encoded = {k: v.squeeze(0) for k, v in encoded.items()}
        return encoded

    def generate_prompt(
        self,
        instruction: str,
        data: str,
        label: str,
    ) -> str:
        prompt = f"""### Instruction:
{instruction} 

### Input:
{data}

### Response:
{label} """.strip()
        return prompt

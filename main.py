import dotenv

dotenv.load_dotenv(
    override=True,
)

import os
import warnings

os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["HF_HOME"] = os.environ.get("HF_HOME")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

import hydra
from omegaconf import DictConfig

from src.pipelines.pipeline import train


@hydra.main(
    config_path="configs/",
    config_name="huggingface.yaml",
)
def main(
    config: DictConfig,
) -> None:
    if config.mode == "train":
        return train(config)
    else:
        raise ValueError(f"Invalid execution mode: {config.mode}")


if __name__ == "__main__":
    main()

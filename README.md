# Korean (s)LLM model fine-tuning pipeline

## For (s)LLM model fine-tuning

### Dataset
HuggingFace Korean dataset(preprocessed as system_prompt, instruction, and output)

### Quick setup

```bash
# clone project
git clone https://github.com/DimensionSTP/llm-fine-tune-hf-tpu.git
cd llm-fine-tune-hf-tpu

# [OPTIONAL] create conda environment
conda create -n myenv python=3.8
conda activate myenv

# install requirements
pip install -r requirements.txt
```

### .env file setting
```shell
PROJECT_DIR={PROJECT_DIR}
CONNECTED_DIR={CONNECTED_DIR}
HF_HOME={HF_HOME}
USER_NAME={USER_NAME}
```

### Training

* end-to-end
```shell
python main.py mode=train is_tuned={tuned or untuned} num_trials={num_trials}
```

### Examples of shell scipts

* train
```shell
bash scripts/train.sh
```

### Additional Options

* model path at HuggingFace Model card
```shell
model_path={model_path}
```

* Set max length for model training and generation
```shell
max_length={max_length} 
```


__If you want to change main config, use --config-name={config_name}.__

__Also, you can use --multirun option.__

__You can set additional arguments through the command line.__
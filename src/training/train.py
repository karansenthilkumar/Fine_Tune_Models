
import yaml
import torch
import mlflow

from transformers import AutoModelForCausalLM
from bitsandbytes import BitsAndBytesConfig
from peft import get_peft_model
from trl import SFTTrainer

from src.data.load_data import load_data
from src.data.preprocess import preprocess
from src.data.tokenizer import get_tokenizer
from src.training.lora_config import get_lora_config
from src.training.trainer_utils import get_training_args

def load_config():
    with open("src/utils/config.yaml") as f:
        return yaml.safe_load(f)

def train():
    config = load_config()
    mlflow.set_experiment(config["mlflow"]["experiment_name"])
    mlflow.start_run()

    model_name = config["model"]["name"]
    tokenizer = get_tokenizer(model_name)

    bnb_config = BitsAndBytesConfig(load_in_4bit=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto"
    )

    dataset = load_data()
    train_data = preprocess(dataset["train"])

    model = get_peft_model(model, get_lora_config())

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_data,
        dataset_text_field="text",
        tokenizer=tokenizer,
        args=get_training_args(config)
    )

    trainer.train()
    model.save_pretrained("./models/lora")

    mlflow.log_param("model", model_name)
    mlflow.end_run()

if __name__ == "__main__":
    train()

from transformers import TrainingArguments

def get_training_args(config):
    return TrainingArguments(
        output_dir="./models",
        per_device_train_batch_size=config["training"]["batch_size"],
        num_train_epochs=config["training"]["epochs"],
        learning_rate=float(config["training"]["lr"]),
        logging_steps=10
    ) 


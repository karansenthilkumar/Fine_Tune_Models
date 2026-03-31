
from datasets import load_dataset

def load_data():
    dataset = load_dataset("json", data_files="data/raw/dataset.json")
    return dataset["train"].train_test_split(test_size=0.1)

from transformers import AutoModelForCausalLM
from peft import PeftModel

def merge():
    base = "mistralai/Mistral-7B-v0.1"
    model = AutoModelForCausalLM.from_pretrained(base)
    model = PeftModel.from_pretrained(model, "./models/lora")

    model = model.merge_and_unload()
    model.save_pretrained("./models/merged")

if __name__ == "__main__":
    merge()


    
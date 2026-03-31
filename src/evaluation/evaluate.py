
from transformers import pipeline

def evaluate():
    pipe = pipeline("text-generation", model="./models/merged")
    print(pipe("Explain ML", max_length=100))

if __name__ == "__main__":
    evaluate() 

    

from fastapi import FastAPI
from src.rag.rag_pipeline import generate_with_rag

app = FastAPI()

@app.get("/generate")
def generate(prompt: str):
    return {"response": generate_with_rag(prompt)} 


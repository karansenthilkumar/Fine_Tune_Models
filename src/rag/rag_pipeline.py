
from vllm import LLM, SamplingParams
from src.rag.retriever import Retriever

docs = [
    "Churn prediction predicts customer churn",
    "Overfitting is memorization of training data"
]

retriever = Retriever(docs)
llm = LLM(model="./models/merged")

def generate_with_rag(query):
    context = retriever.retrieve(query)
    prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"

    params = SamplingParams(max_tokens=200)
    out = llm.generate([prompt], params)

    return out[0].outputs[0].text


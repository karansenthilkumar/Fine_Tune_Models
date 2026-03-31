
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class Retriever:
    def __init__(self, docs):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.docs = docs
        emb = self.model.encode(docs)
        self.index = faiss.IndexFlatL2(emb.shape[1])
        self.index.add(np.array(emb))

    def retrieve(self, query):
        q = self.model.encode([query])
        _, idx = self.index.search(np.array(q), 2)
        return [self.docs[i] for i in idx[0]]
    

    
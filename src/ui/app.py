

import gradio as gr
from src.rag.rag_pipeline import generate_with_rag

gr.Interface(
    fn=generate_with_rag,
    inputs="text",
    outputs="text",
    title="LLM + RAG"
).launch()
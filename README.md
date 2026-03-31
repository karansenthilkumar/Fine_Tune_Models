
#  LLM Fine-Tuning Pipeline (QLoRA + RAG + vLLM)

##  Overview

This project implements a **production-grade LLM fine-tuning pipeline** using QLoRA for efficient training and RAG for improved response accuracy.

---

##  Features

*  Fine-tuning 7B LLM (Mistral) using QLoRA (4-bit)
*  LoRA adapters via PEFT
*  Experiment tracking with MLflow
*  RAG pipeline using FAISS
*  High-performance inference using vLLM
*  FastAPI deployment
*  Gradio UI demo

---

##  Architecture

Data → Preprocessing → QLoRA Fine-tuning → MLflow Tracking
→ Model Merge → RAG → vLLM → FastAPI → UI

---

##  Tech Stack

* Transformers
* PEFT
* bitsandbytes
* MLflow
* FAISS
* FastAPI
* vLLM
* Gradio

---

## ▶ How to Run

### 1. Install dependencies

pip install -r requirements.txt

### 2. Train model

python src/training/train.py

### 3. Merge model

python src/training/merge_model.py

### 4. Run API

uvicorn src.serving.api:app --reload

### 5. Run UI

python src/ui/app.py

---

##  Deployment

Deployed on AWS EC2 using Docker.

---

##  Use Cases

* Customer support chatbot
* Telecom churn prediction assistant
* Domain-specific knowledge assistant

---

##  Results

* Reduced training cost by ~70% using QLoRA
* Improved response accuracy using RAG

------------




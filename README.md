# QuantSense

QuantSense is an Agentic Financial Intelligence System built for the IIT Kharagpur RAG & Agentic AI Hackathon.

## Features

* Hybrid Retrieval (BM25 + Dense Retrieval)
* HyDE Query Expansion
* CRAG Self-Correction
* Financial Guardrails
* Confidence Scoring
* Management Tone Drift Detection
* Streamlit Interface
* RAG Evaluation Pipeline

## Tech Stack

* Python
* LangGraph
* ChromaDB
* Sentence Transformers
* Groq LLM
* Streamlit

## Running Locally

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

## Project Structure

* `src/agent` – LangGraph workflow
* `src/retrieval` – Retrieval and HyDE
* `src/guardrails` – Hallucination and citation checks
* `src/features` – Confidence scoring and FinSentinel
* `app` – Streamlit UI

Built for IIT KGP Arch Hackathon 2026.

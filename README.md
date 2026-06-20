# QuantSense

Agentic Financial Intelligence Platform built for IIT KGP Arch Hackathon 2026.

## Live Demo

https://quantsense-rag-9aomprxaxh2appphvvh8bzg.streamlit.app

## GitHub Repository

https://github.com/Aritra0401/quantsense-rag

---

## Problem Statement

Financial reports, earnings transcripts, and filings contain large amounts of unstructured information.

Analysts spend significant time searching documents, validating information, and avoiding hallucinated answers from LLMs.

QuantSense provides an agentic RAG pipeline that retrieves, verifies, and synthesizes financial information from company documents.

## Features

- Hybrid Retrieval (BM25 + ChromaDB)
- HyDE Query Expansion
- Reciprocal Rank Fusion (RRF)
- Cross Encoder Reranking
- CRAG Self Correction
- Hallucination Detection
- Financial Intelligence Assistant
- Management Tone Drift Analysis
- Streamlit Web Interface

## Architecture

User Query
    ↓
HyDE Query Expansion
    ↓
BM25 Retrieval + ChromaDB Retrieval
    ↓
Reciprocal Rank Fusion
    ↓
Cross Encoder Reranking
    ↓
CRAG Verification
    ↓
Hallucination Guard
    ↓
LLM Response


## Tech Stack

- Python
- Streamlit
- ChromaDB
- BM25
- Sentence Transformers
- LangGraph
- Groq
- Tavily Search
- Docker


## Project Structure

src/
├── agent/
├── ingestion/
├── retrieval/
├── features/
├── guardrails/

data/
├── raw/
├── processed/

app/
└── streamlit_app.py

## Local Setup

1. Clone repository

git clone https://github.com/Aritra0401/quantsense-rag.git

2. Create virtual environment

python -m venv venv

3. Install requirements

pip install -r requirements.txt

4. Create .env

GROQ_API_KEY=YOUR_KEY
TAVILY_API_KEY=YOUR_KEY

5. Run application

streamlit run app/streamlit_app.py


## Example Queries

- What is Apple's P/E ratio?
- Summarize Infosys FY2026 annual report.
- Compare Apple and JPMorgan valuation.
- What were the major risks highlighted in recent filings?

## Hackathon Submission

IIT KGP Arch Hackathon 2026

Submission Includes:
- Production Code (GitHub)
- Live Demo (Streamlit Cloud)
- Architecture & Case Study PPT
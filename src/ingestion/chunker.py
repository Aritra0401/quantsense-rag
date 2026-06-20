# src/ingestion/chunker.py
# Parses HTML/TXT files and chunks them with rich metadata
import os
import json
import re

from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger
from pypdf import PdfReader

SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=64,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def clean_html(html_text: str) -> str:
    soup = BeautifulSoup(html_text, "html.parser")

    for tag in soup(["script", "style", "table"]):
        tag.decompose()

    text = soup.get_text(separator=" ")
    text = re.sub(r"\s+", " ", text).strip()

    return text


def read_file(filepath: str) -> str:
    """
    Read content from PDF, HTML, or TXT files.
    """

    if filepath.endswith(".pdf"):
        reader = PdfReader(filepath)

        text = ""

        for page in reader.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

        return text

    elif filepath.endswith(".htm") or filepath.endswith(".html"):
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            raw = f.read()

        return clean_html(raw)

    else:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            return f.read()


def chunk_file(filepath: str,
               out_dir: str = "data/processed/chunks") -> list:

    fname = os.path.basename(filepath)

    try:
        parts = fname.rsplit(".", 1)[0].split("_")

        ticker = parts[0]
        form_type = parts[1]
        date = parts[2]

    except Exception:
        logger.warning(f"Skipping improperly named file: {fname}")
        return []

    text = read_file(filepath)

    chunks = SPLITTER.split_text(text)

    docs = []

    for i, chunk in enumerate(chunks):

        if len(chunk.strip()) < 50:
            continue

        docs.append(
            {
                "id": f"{ticker}_{form_type}_{date}_{i}",
                "text": chunk,
                "metadata": {
                    "ticker": ticker,
                    "form_type": form_type,
                    "date": date,
                    "chunk_index": i,
                    "source_file": fname,
                },
            }
        )

    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(
        out_dir,
        f"{ticker}_{form_type}_{date}.json"
    )

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(docs, f, indent=2)

    logger.info(f"Chunked {len(docs)} chunks from {fname}")

    return docs


def chunk_all_filings(
    filings_dir: str = "data/raw/filings"
):

    all_docs = []

    if not os.path.exists(filings_dir):
        logger.error(f"Directory not found: {filings_dir}")
        return []

    supported_extensions = (
        ".htm",
        ".html",
        ".txt",
        ".pdf",
    )

    for fname in os.listdir(filings_dir):

        if not fname.lower().endswith(supported_extensions):
            continue

        fpath = os.path.join(filings_dir, fname)

        all_docs.extend(chunk_file(fpath))

    logger.info(f"Total chunks: {len(all_docs)}")

    return all_docs


if __name__ == "__main__":
    chunk_all_filings()
"""Download, parse, chunk, embed and persist the two official NIST documents."""
from __future__ import annotations

import argparse
from pathlib import Path
import re

import chromadb
import pymupdf
import requests
from bs4 import BeautifulSoup
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import (
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
)

RAW_DIR = Path("data/raw")
AI_RMF_URL = "https://nvlpubs.nist.gov/nistpubs/ai/nist.ai.100-1.pdf"
GENAI_PROFILE_PAGE = (
    "https://www.nist.gov/publications/"
    "artificial-intelligence-risk-management-framework-generative-artificial-intelligence"
)
GENAI_PROFILE_PDF_FALLBACK = "https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf"


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=90, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()
    destination.write_bytes(response.content)
    if not destination.read_bytes().startswith(b"%PDF"):
        raise ValueError(f"Downloaded content is not a PDF: {url}")


def find_profile_pdf() -> str:
    """Find the PDF linked from the official NIST publication page."""
    try:
        response = requests.get(
            GENAI_PROFILE_PAGE, timeout=60, headers={"User-Agent": "Mozilla/5.0"}
        )
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        candidates = []
        for anchor in soup.find_all("a", href=True):
            href = anchor["href"]
            label = anchor.get_text(" ", strip=True).lower()
            full_url = requests.compat.urljoin(GENAI_PROFILE_PAGE, href)
            if ".pdf" in href.lower() or "download" in label or "pdf" in label:
                candidates.append(full_url)

        for url in candidates:
            if "generative" in url.lower() or "genai" in url.lower() or "ai.600" in url.lower():
                return url
        for url in candidates:
            if ".pdf" in url.lower():
                return url
    except requests.RequestException:
        pass
    return GENAI_PROFILE_PDF_FALLBACK


def detect_section(text: str) -> str:
    """Best-effort section heading detection for provenance metadata."""
    patterns = [
        r"^\s*(\d+(?:\.\d+){0,3})\s+([A-Z][^\n]{2,100})$",
        r"^\s*(Appendix\s+[A-Z][^\n]{0,100})$",
    ]
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in lines:
        for pattern in patterns:
            match = re.match(pattern, line)
            if match:
                return " ".join(match.groups()).strip()
    return "Not detected"


def pdf_to_documents(pdf_path: Path, document_name: str) -> list[Document]:
    pdf = pymupdf.open(pdf_path)
    docs = []
    current_section = "Not detected"

    for page_index, page in enumerate(pdf):
        page_text = page.get_text("text").strip()
        if not page_text:
            continue

        detected = detect_section(page_text)
        if detected != "Not detected":
            current_section = detected

        docs.append(
            Document(
                page_content=page_text,
                metadata={
                    "source": document_name,
                    "page": page_index + 1,
                    "section": current_section,
                    "file_name": pdf_path.name,
                },
            )
        )
    pdf.close()
    return docs


def _collection_has_chunks() -> bool:
    try:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        collection = client.get_collection(COLLECTION_NAME)
        return collection.count() > 0
    except Exception:
        return False


def ingest(force: bool = False) -> None:
    if not force and _collection_has_chunks():
        print("Existing NIST vector database found. Skipping ingestion.")
        return

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    ai_pdf = RAW_DIR / "nist_ai_rmf_1_0.pdf"
    genai_pdf = RAW_DIR / "nist_genai_profile.pdf"

    print("Downloading NIST AI RMF 1.0...")
    if not ai_pdf.exists():
        download_file(AI_RMF_URL, ai_pdf)

    print("Finding/downloading NIST Generative AI Profile...")
    if not genai_pdf.exists():
        download_file(find_profile_pdf(), genai_pdf)

    page_documents = (
        pdf_to_documents(ai_pdf, "NIST AI RMF 1.0")
        + pdf_to_documents(genai_pdf, "NIST AI RMF Generative AI Profile")
    )
    if not page_documents:
        raise RuntimeError("No text could be extracted from the NIST PDFs.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(page_documents)

    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"nist_chunk_{index}"
        chunk.metadata["start_index"] = chunk.metadata.get("start_index", 0)

    print(f"Creating embeddings for {len(chunks)} chunks...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    try:
        existing = Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings,
        )
        existing.delete_collection()
    except Exception:
        pass

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
    )
    print(f"Done. Indexed {len(chunks)} chunks in {CHROMA_DIR}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest NIST AI RMF documents.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild the vector store even if it already exists.",
    )
    ingest(force=parser.parse_args().force)

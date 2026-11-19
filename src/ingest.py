"""Download, parse, chunk, embed and persist the two official NIST documents."""
from pathlib import Path
import re
import requests
from bs4 import BeautifulSoup
import fitz
import chromadb

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

client = chromadb.PersistentClient(path=CHROMA_DIR)

try:
    collection = client.get_collection(COLLECTION_NAME)

    if collection.count() > 0:
        print("Existing NIST vector database found. Skipping ingestion.")
        return

except Exception:
    pass

from src.config import (
    CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL,
    CHUNK_SIZE, CHUNK_OVERLAP,
)

RAW_DIR = Path("data/raw")
AI_RMF_URL = "https://nvlpubs.nist.gov/nistpubs/ai/nist.ai.100-1.pdf"
GENAI_PROFILE_PAGE = (
    "https://www.nist.gov/publications/artificial-intelligence-risk-management-framework-generative-artificial-intelligence"
)

def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=90, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()
    destination.write_bytes(response.content)
    if not destination.read_bytes().startswith(b"%PDF"):
        raise ValueError(f"Downloaded content is not a PDF: {url}")

def find_profile_pdf() -> str:
    """Find the PDF linked from the official NIST publication page."""
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

    # Prefer links whose URL or link text identifies the Generative AI profile.
    for url in candidates:
        if "generative" in url.lower() or "genai" in url.lower():
            return url
    for url in candidates:
        if ".pdf" in url.lower():
            return url
    raise RuntimeError(
        "Could not find a PDF link on the official NIST GenAI Profile page. "
        "Open the NIST page and download the profile PDF into data/raw/genai_profile.pdf."
    )

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
    pdf = fitz.open(pdf_path)
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

def ingest() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    ai_pdf = RAW_DIR / "nist_ai_rmf_1_0.pdf"
    genai_pdf = RAW_DIR / "nist_genai_profile.pdf"

    print("Downloading NIST AI RMF 1.0...")
    if not ai_pdf.exists():
        download_file(AI_RMF_URL, ai_pdf)

    print("Finding/downloading NIST Generative AI Profile...")
    if not genai_pdf.exists():
        profile_url = find_profile_pdf()
        download_file(profile_url, genai_pdf)

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

    # Add a stable chunk identifier and preserve page/section/source metadata.
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"nist_chunk_{index}"
        chunk.metadata["start_index"] = chunk.metadata.get("start_index", 0)

    print(f"Creating embeddings for {len(chunks)} chunks...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # Rebuild the collection so rerunning ingestion does not duplicate chunks.
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
    ingest()

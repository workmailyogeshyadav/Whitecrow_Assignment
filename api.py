
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Build/load the vector database before accepting requests.
    from src.ingest import ingest

    print("Initializing NIST knowledge base...")
    ingest()
    print("Knowledge base ready.")

    yield


app = FastAPI(
    title="NIST Agentic RAG API",
    description="Conversational RAG using LangGraph and NIST AI RMF documents",
    version="1.0.0",
    lifespan=lifespan,
)


class ChatRequest(BaseModel):
    question: str
    chat_history: list[dict[str, str]] = Field(default_factory=list)


@app.get("/")
def home():
    return {
        "message": "NIST Agentic RAG API is running",
        "docs": "/docs"
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/ask")
def ask_question(request: ChatRequest):
    try:
        # Import after ingestion has completed.
        from src.graph import ask

        result = ask(
            question=request.question,
            chat_history=request.chat_history
        )

        return {
            "question": request.question,
            "answer": result.get("answer", ""),
            "sources": result.get("sources", [])
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process question: {str(exc)}"
        )

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    from src.ingest import ingest

    print("Initializing NIST knowledge base...")
    ingest()
    print("Knowledge base ready.")
    yield


app = FastAPI(
    title="NIST Agentic RAG",
    description="Conversational RAG using LangGraph and NIST AI RMF documents",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class ChatRequest(BaseModel):
    question: str
    chat_history: list[dict[str, str]] = Field(default_factory=list)


@app.get("/")
def home():
    index = STATIC_DIR / "index.html"
    if index.exists():
        return FileResponse(index)
    return {"message": "NIST Agentic RAG API is running", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/ask")
def ask_question(request: ChatRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        from src.graph import ask

        result = ask(question=question, chat_history=request.chat_history)
        return {
            "question": question,
            "standalone_question": result.get("standalone_question", question),
            "answer": result.get("answer", ""),
            "sources": result.get("sources", []),
            "retrieval_sufficient": result.get("retrieval_sufficient", False),
            "retry_count": result.get("retry_count", 0),
            "failure_reason": result.get("failure_reason", ""),
        }
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process question: {str(exc)}",
        )

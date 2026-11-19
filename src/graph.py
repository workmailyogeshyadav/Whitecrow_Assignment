"""LangGraph RAG workflow: understand -> retrieve -> grade -> answer/retry."""
from typing import TypedDict, Any
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

from src.config import (
    GROQ_API_KEY, GROQ_MODEL, EMBEDDING_MODEL, CHROMA_DIR,
    COLLECTION_NAME, TOP_K, MIN_RELEVANCE_SCORE, MAX_RETRIES,
)

class RAGState(TypedDict, total=False):
    question: str
    standalone_question: str
    chat_history: list[dict[str, str]]
    documents: list[Any]
    context: str
    answer: str
    sources: list[dict[str, Any]]
    retry_count: int
    retrieval_sufficient: bool
    failure_reason: str

def build_components():
    if not GROQ_API_KEY:
        raise EnvironmentError("GROQ_API_KEY is missing. Add it to your .env file.")
    llm = ChatGroq(
        model=GROQ_MODEL,
        temperature=0,
        api_key=GROQ_API_KEY,
    )
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )
    return llm, vectorstore

llm, vectorstore = build_components()

def understand_query(state: RAGState) -> dict:
    """Resolve pronouns and implicit references in follow-up questions."""
    question = state["question"]
    history = state.get("chat_history", [])
    if not history:
        return {"standalone_question": question}

    history_text = "\n".join(
        f'{item["role"].upper()}: {item["content"]}' for item in history[-6:]
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Rewrite the latest user question as a standalone search query. "
         "Use conversation history only to resolve references such as 'which one', "
         "'why', or 'tell me more'. Preserve the user's intent. "
         "Return only the rewritten question."),
        ("human", "Conversation:\n{history}\n\nLatest question: {question}")
    ])
    result = (prompt | llm).invoke(
        {"history": history_text, "question": question}
    )
    return {"standalone_question": result.content.strip()}

def retrieve(state: RAGState) -> dict:
    query = state.get("standalone_question", state["question"])
    results = vectorstore.similarity_search_with_relevance_scores(query, k=TOP_K)

    docs = []
    for doc, score in results:
        # Chroma relevance scores are normalized by the vector-store integration.
        doc.metadata["relevance_score"] = float(score)
        docs.append(doc)

    context_parts = []
    sources = []
    seen = set()
    for doc in docs:
        meta = doc.metadata
        source_key = (
            meta.get("source"), meta.get("page"), meta.get("section")
        )
        if source_key not in seen:
            seen.add(source_key)
            sources.append({
                "document": meta.get("source", "Unknown document"),
                "page": meta.get("page", "Unknown"),
                "section": meta.get("section", "Not detected"),
            })
        context_parts.append(
            f"[SOURCE: {meta.get('source', 'Unknown')}; "
            f"PAGE: {meta.get('page', 'Unknown')}; "
            f"SECTION: {meta.get('section', 'Not detected')}; "
            f"RELEVANCE: {meta.get('relevance_score', 0):.3f}]\n"
            f"{doc.page_content}"
        )

    return {
        "documents": docs,
        "context": "\n\n---\n\n".join(context_parts),
        "sources": sources,
    }

def grade_retrieval(state: RAGState) -> dict:
    docs = state.get("documents", [])
    best_score = max(
        (doc.metadata.get("relevance_score", 0.0) for doc in docs),
        default=0.0,
    )
    sufficient = bool(docs) and best_score >= MIN_RELEVANCE_SCORE
    return {
        "retrieval_sufficient": sufficient,
        "failure_reason": "" if sufficient else (
            f"No sufficiently relevant evidence found (best score: {best_score:.3f})."
        ),
    }

def route_after_grading(state: RAGState) -> str:
    if state.get("retrieval_sufficient"):
        return "answer"
    if state.get("retry_count", 0) < MAX_RETRIES:
        return "rewrite"
    return "insufficient"

def rewrite_query(state: RAGState) -> dict:
    """Broaden/rephrase a weak query once, without inventing new facts."""
    original = state.get("standalone_question", state["question"])
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Create one alternative search query for a vector database containing "
         "the NIST AI Risk Management Framework and its Generative AI Profile. "
         "Use related terminology and expand abbreviations where useful. "
         "Do not answer the question. Return only the query."),
        ("human", "Original query: {question}")
    ])
    result = (prompt | llm).invoke({"question": original})
    return {
        "standalone_question": result.content.strip(),
        "retry_count": state.get("retry_count", 0) + 1,
    }

def generate_answer(state: RAGState) -> dict:
    question = state.get("question", "")
    standalone = state.get("standalone_question", question)
    history = state.get("chat_history", [])
    history_text = "None"
    if history:
        history_text = "\n".join(
            f'{item["role"].upper()}: {item["content"]}' for item in history[-6:]
        )

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a careful assistant answering questions about the NIST AI RMF. "
         "Answer ONLY from the supplied retrieved context. Do not use outside "
         "knowledge to fill gaps. If evidence is partial, state the limitation. "
         "Use conversation history only to interpret follow-up questions. "
         "Cite claims inline using [Document, p. X, Section Y]. "
         "Never invent page numbers, sections, or sources. If the context does "
         "not support an answer, say so."),
        ("human",
         "Conversation history:\n{history}\n\n"
         "Retrieved context:\n{context}\n\n"
         "Latest user question: {question}\n"
         "Resolved question: {standalone}\n\n"
         "Write a clear, concise answer with citations.")
    ])
    result = (prompt | llm).invoke({
        "context": state.get("context", ""),
        "question": question,
        "standalone": standalone,
        "history": history_text,
    })
    return {"answer": result.content}

def insufficient_context(state: RAGState) -> dict:
    return {
        "answer": (
            "I couldn't find sufficiently relevant evidence in the indexed NIST "
            "documents to answer this confidently. Please rephrase the question "
            "or ask about a specific AI RMF concept."
        ),
        "sources": [],
    }

def create_graph():
    builder = StateGraph(RAGState)
    builder.add_node("understand_query", understand_query)
    builder.add_node("retrieve", retrieve)
    builder.add_node("grade_retrieval", grade_retrieval)
    builder.add_node("rewrite_query", rewrite_query)
    builder.add_node("generate_answer", generate_answer)
    builder.add_node("insufficient_context", insufficient_context)

    builder.set_entry_point("understand_query")
    builder.add_edge("understand_query", "retrieve")
    builder.add_edge("retrieve", "grade_retrieval")
    builder.add_conditional_edges(
        "grade_retrieval",
        route_after_grading,
        {
            "answer": "generate_answer",
            "rewrite": "rewrite_query",
            "insufficient": "insufficient_context",
        },
    )
    builder.add_edge("rewrite_query", "retrieve")
    builder.add_edge("generate_answer", END)
    builder.add_edge("insufficient_context", END)
    return builder.compile()

rag_graph = create_graph()

def ask(question: str, chat_history: list[dict[str, str]] | None = None) -> dict:
    result = rag_graph.invoke({
        "question": question,
        "chat_history": chat_history or [],
        "retry_count": 0,
    })
    return result

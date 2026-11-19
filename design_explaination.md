# Design Explanation — Agentic RAG System Using LangGraph

## 1. Objective

This application answers natural-language questions using two official NIST knowledge sources:

1. NIST AI Risk Management Framework (AI RMF 1.0)
2. NIST AI RMF: Generative Artificial Intelligence Profile

The solution uses retrieval-augmented generation (RAG) so answers are grounded in retrieved document passages. LangGraph orchestrates query understanding, retrieval, relevance checking, query rewriting, and answer generation.

The implementation is intentionally small and focused on the assignment requirements rather than production-scale infrastructure.

## 2. High-level architecture

### Document ingestion

```text
Official NIST PDF sources
        |
        v
Download documents
        |
        v
Extract text page by page (PyMuPDF)
        |
        v
Split pages into overlapping chunks
        |
        v
Generate embeddings (Sentence Transformers)
        |
        v
Persist vectors and metadata (ChromaDB)
```

### Question answering

```text
User question + chat history
        |
        v
Understand / rewrite follow-up query
        |
        v
Retrieve top-k chunks from ChromaDB
        |
        v
Grade retrieval relevance
        |
        +------------------------+
        |                        |
   Sufficient                Insufficient
        |                        |
        v                        v
Generate grounded          Rewrite query
answer with citations           |
        |                       v
        |                  Retrieve again
        |                       |
        |                 Grade retrieval
        |                       |
        |               Still insufficient
        |                       |
        |                       v
        |                  Abstain safely
        v
Answer + source metadata
```

## 3. Technology choices

| Component | Technology | Reason |
|---|---|---|
| Language | Python | Required by the assignment and mature RAG ecosystem |
| Orchestration | LangGraph | Stateful nodes and conditional workflow edges |
| PDF extraction | PyMuPDF | Page-level text extraction and provenance |
| Chunking | RecursiveCharacterTextSplitter | Splits using paragraph and line boundaries where possible |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 | Lightweight local embedding model |
| Vector store | ChromaDB | Simple persistent local vector database |
| LLM | Groq-hosted chat model | API-based language-model inference |
| API | FastAPI | HTTP endpoints and interactive Swagger documentation |

These are pragmatic choices for a small assignment, not claims that they are universally optimal.

## 4. Document ingestion

The ingestion pipeline downloads both NIST documents and extracts text one PDF page at a time.

Each page is represented as a document with metadata:

- `source`: human-readable document name
- `page`: one-based PDF page number
- `section`: best-effort detected section heading
- `file_name`: original local PDF filename

The page text is split using `RecursiveCharacterTextSplitter` with a chunk size of **900 characters** and overlap of **150 characters**. Overlap retains some context between adjacent chunks. Page-first splitting makes page attribution straightforward and avoids chunks silently spanning multiple PDF pages.

Section detection uses simple heading-pattern matching. It is heuristic: some pages may have `Not detected`, and a previously detected section may carry forward. It is not a complete PDF structure parser.

The embedding model converts each chunk into a numerical vector. ChromaDB stores the vectors together with their text and metadata.

Ingestion is separate from question answering. It should run when the knowledge base is first created or intentionally rebuilt, not for every question.

## 5. Retrieval strategy

For each question, the application searches ChromaDB for the **top 5** most similar chunks.

The embedding model also embeds the search query. ChromaDB compares the query vector with stored chunk vectors and returns the closest matches with relevance scores.

The retrieved context includes chunk text and source metadata. This information is passed to the answer-generation node.

The relevance threshold is configurable through `MIN_RELEVANCE_SCORE`. The current implementation treats retrieval as sufficient when at least one returned chunk reaches the configured threshold.

This threshold is a lightweight heuristic, not a trained relevance classifier. Its value should be calibrated against representative relevant and irrelevant questions. Similarity scores vary with the embedding model, vector-store integration, and corpus, so the default threshold should be validated.

## 6. LangGraph workflow

The RAG workflow is implemented as a LangGraph `StateGraph`. The graph state carries the question, standalone query, conversation history, retrieved documents, context, answer, sources, retry count, and retrieval decision.

### Nodes

**`understand_query`**  
Uses recent conversation history to rewrite a follow-up question as a standalone search query. For example, after “What are the four functions of the NIST AI RMF?”, the follow-up “Which one is cross-cutting?” can be rewritten as “Which of the four NIST AI RMF core functions is cross-cutting?”

**`retrieve`**  
Performs similarity search against ChromaDB and retrieves the top-k chunks. It assembles context for the LLM and collects source metadata.

**`grade_retrieval`**  
Checks returned relevance scores against the configured threshold. Sufficient results proceed to answer generation. Weak results can trigger one retry. This is a score-based gate, not an LLM-based semantic grader.

**`rewrite_query`**  
Asks the LLM to produce an alternative search query. The rewritten query is sent through retrieval again. The retry is bounded by `MAX_RETRIES` to prevent an infinite loop.

**`generate_answer`**  
Passes the original user question and retrieved context to the LLM. The prompt instructs the model to answer only from the supplied context, state limitations when evidence is incomplete, and cite source document, page, and section.

**`insufficient_context`**  
If retrieval remains insufficient after the retry, the graph returns a fallback response rather than attempting to answer from general model knowledge.

### Conditional routing

After `grade_retrieval`, LangGraph selects one of three paths:

- Sufficient evidence → `generate_answer`
- Insufficient evidence with retry available → `rewrite_query`
- Insufficient evidence after retry → `insufficient_context`

This conditional flow is the agentic element: the next action depends on the retrieval decision held in graph state.

## 7. Conversational context

The API accepts `chat_history` in each `/ask` request. The query-understanding node uses recent turns to resolve follow-up questions.

The current API is stateless between HTTP requests. The client must send previous user and assistant messages with each follow-up request. The CLI example maintains rolling in-memory history, but it is lost when the process restarts.

For a longer-lived multi-user application, conversation state could be stored using a LangGraph checkpointer and a unique thread ID per conversation.

## 8. Source attribution

The source document, page number, and detected section are stored in each chunk's metadata. The retrieval node collects these values from the chunks it returns.

The API returns a `sources` array alongside the answer. The prompt also asks the LLM to use inline citations.

The source list is derived from retrieved metadata. Inline citations are generated by the LLM and are not independently validated in this basic implementation. A stronger implementation would validate each cited page and section against the retrieved chunks before returning the answer.

## 9. Insufficient-context and failure handling

The system uses several safeguards:

1. If no useful chunks are retrieved, the relevance gate can reject the context.
2. If retrieval is weak, the graph rewrites the query and retries once.
3. If the retry is still insufficient, the graph returns an abstention response.
4. The answer prompt instructs the LLM not to fill evidence gaps using outside knowledge.
5. The API catches request-processing exceptions and returns an HTTP error response.

The relevance threshold is a heuristic. It does not guarantee that every accepted retrieval is relevant or every rejected retrieval is irrelevant. This should be evaluated with a test set.

## 10. Running and deployment

For Render, configure:

- Python version: 3.12
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn api:app --host 0.0.0.0 --port $PORT`
- Health-check path: `/health`

The FastAPI lifespan hook initializes the knowledge base before serving requests. If the ChromaDB collection already exists and contains chunks, ingestion can be skipped.

**Persistence note:** Render's default filesystem is ephemeral. Without a persistent disk or external vector database, ChromaDB data may be lost when an instance is replaced or redeployed. In that case, the application must rebuild the index during startup. For persistence across deployments, mount a disk and set `CHROMA_DIR` to a directory on that disk.

The embedding model may need to be downloaded on first startup, increasing startup time and memory use.

## 11. Known limitations and possible improvements

- Section extraction is heuristic and may not identify every section correctly.
- Fixed-size character chunking does not fully understand document structure.
- Similarity thresholding is a simple retrieval-quality check, not a semantic relevance model.
- Only one query rewrite retry is performed.
- Conversation history is supplied by the client and is not persisted by the API.
- Inline citations generated by the LLM are not independently verified.
- There is no OCR pipeline for scanned PDFs.
- The application does not include authentication, rate limiting, multi-user isolation, or a frontend.
- There is no automated benchmark suite in the basic implementation.

Possible improvements include hybrid keyword/vector retrieval, MMR, reranking, an LLM-based relevance grader, citation validation, automated retrieval and answer evaluation, and persistent per-user conversation state.

## 12. Evaluation plan

Test the application with:

1. A direct question about the four NIST AI RMF functions.
2. A question about the NIST Generative AI Profile.
3. At least two follow-up questions that depend on prior conversation context.
4. An unrelated question to verify abstention.
5. A question whose answer requires checking the source page and section.

Useful future metrics include Recall@K, Precision@K, Mean Reciprocal Rank (MRR), answer relevance, and faithfulness. These should be measured using prepared questions and reference evidence rather than inferred from a few manual demonstrations.

## 13. Summary

The implementation separates document ingestion from question answering, stores source metadata alongside vector embeddings, and uses LangGraph to control retrieval, relevance checking, query rewriting, and answer generation. Conversation history supports follow-up questions, while a bounded retry and abstention path reduce the chance of answering when retrieved NIST evidence is insufficient.

The design prioritizes clarity, traceable sources, and a manageable implementation appropriate to the assignment's stated scope.

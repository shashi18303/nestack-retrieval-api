#!/usr/bin/env python3
"""
server.py - Part 2: Semantic Retrieval API
FastAPI server exposing endpoints to search ingested vector chunks from ChromaDB.

IMPORTANT: This is a semantic search system, not a chatbot.
The retrieval endpoint DOES NOT call an LLM to generate an answer.
It returns the original text chunks found by vector similarity.
"""

import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Query, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import chromadb
from chromadb.utils import embedding_functions

# Configuration
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "pdf_documents")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "3"))

# Initialize FastAPI App
app = FastAPI(
    title="Nestack PDF Semantic Retrieval Engine",
    description="Vector similarity retrieval system for PDF chunks using ChromaDB and Sentence Transformers. No LLM generation.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_collection = None


def get_collection():
    """
    Lazy loads and returns the persistent ChromaDB collection.
    Uses SentenceTransformerEmbeddingFunction for automatic query vectorization.
    """
    global _collection
    if _collection is None:
        if not os.path.exists(CHROMA_PERSIST_DIR):
            raise RuntimeError(
                f"ChromaDB directory '{CHROMA_PERSIST_DIR}' does not exist. "
                "Please run 'python ingest.py' first to ingest document.pdf."
            )

        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL_NAME
        )

        try:
            _collection = client.get_collection(
                name=COLLECTION_NAME,
                embedding_function=sentence_transformer_ef
            )
        except Exception as e:
            raise RuntimeError(
                f"Collection '{COLLECTION_NAME}' not found in {CHROMA_PERSIST_DIR}. "
                "Please execute 'python ingest.py' first."
            ) from e

    return _collection


class QueryRequest(BaseModel):
    query: str = Field(..., description="The search query text to embed and search", min_length=1)
    top_k: Optional[int] = Field(default=DEFAULT_TOP_K, ge=1, le=20, description="Number of chunks to return")


class ChunkResult(BaseModel):
    chunk_text: str = Field(..., description="Original verbatim text chunk from the document")
    page_number: int = Field(..., description="1-indexed PDF page number containing the chunk")
    score: float = Field(..., description="Cosine similarity score (higher is more relevant)")


class QueryResponse(BaseModel):
    query: str
    results: List[ChunkResult]
    total_results: int


@app.get("/")
def root():
    """Service health and metadata check."""
    status = "ready"
    chunk_count = 0
    try:
        col = get_collection()
        chunk_count = col.count()
    except Exception as e:
        status = f"uninitialized: {str(e)}"

    return {
        "status": status,
        "engine": "FastAPI + ChromaDB + SentenceTransformers",
        "model": EMBEDDING_MODEL_NAME,
        "vector_count": chunk_count,
        "endpoints": {
            "search_post": "POST /query",
            "search_get": "GET /query?q=<query_text>",
            "health": "GET /health"
        }
    }


@app.get("/health")
def health_check():
    """Returns database readiness."""
    try:
        col = get_collection()
        return {"status": "healthy", "vectors_indexed": col.count()}
    except Exception as e:
        return {"status": "degraded", "detail": str(e)}


def perform_vector_search(query_text: str, top_k: int) -> List[ChunkResult]:
    """
    Executes semantic search:
    1. Embeds the query using the configured Sentence Transformer
    2. Searches the ChromaDB collection using cosine similarity
    3. Formats results to strict schema: chunk_text, page_number, score
    NOTE: Strictly NO LLM calls. Returns original text chunks only.
    """
    collection = get_collection()

    if collection.count() == 0:
        return []

    chroma_res = collection.query(
        query_texts=[query_text],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"]
    )

    documents = chroma_res.get("documents", [[]])[0]
    metadatas = chroma_res.get("metadatas", [[]])[0]
    distances = chroma_res.get("distances", [[]])[0]

    results: List[ChunkResult] = []
    for doc, meta, dist in zip(documents, metadatas, distances):
        # In Chroma with cosine metric, distance = 1 - cosine_similarity.
        # Cosine similarity score = 1 - distance
        similarity_score = max(0.0, 1.0 - float(dist)) if dist is not None else 0.0

        results.append(
            ChunkResult(
                chunk_text=doc,
                page_number=int(meta.get("page_number", 1)),
                score=round(similarity_score, 4)
            )
        )

    return results


@app.post("/query", response_model=QueryResponse)
def query_post(payload: QueryRequest):
    """
    Part 2 Retrieval Endpoint (POST).
    Receives JSON with query string, retrieves top matching chunks from ChromaDB,
    and returns exact JSON structure with chunk_text, page_number, and similarity score.
    """
    try:
        results = perform_vector_search(payload.query, payload.top_k or DEFAULT_TOP_K)
        return QueryResponse(
            query=payload.query,
            results=results,
            total_results=len(results)
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.get("/query", response_model=QueryResponse)
def query_get(
    q: str = Query(..., description="Search query string"),
    top_k: int = Query(DEFAULT_TOP_K, ge=1, le=20, description="Top K items to return")
):
    """
    Part 2 Retrieval Endpoint (GET convenience endpoint).
    """
    try:
        results = perform_vector_search(q, top_k)
        return QueryResponse(
            query=q,
            results=results,
            total_results=len(results)
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload and ingest any new PDF file directly into ChromaDB.
    Extracts pages via PyMuPDF, chunks text, and embeds vectors.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    save_path = os.path.join(os.path.dirname(__file__), "uploaded_" + file.filename)
    contents = await file.read()
    with open(save_path, "wb") as f:
        f.write(contents)

    try:
        from ingest import extract_pages_from_pdf, process_and_chunk_document, store_chunks_in_chromadb, get_chroma_collection
        pages = extract_pages_from_pdf(save_path)
        chunks = process_and_chunk_document(pages)
        collection = get_chroma_collection()
        total_stored = store_chunks_in_chromadb(chunks, collection)
        return {
            "status": "success",
            "filename": file.filename,
            "pages_extracted": len(pages),
            "chunks_indexed": total_stored,
            "total_vectors_in_db": collection.count()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    print(f"Starting Nestack Retrieval API on http://{host}:{port}")
    uvicorn.run("server:app", host=host, port=port, reload=True)

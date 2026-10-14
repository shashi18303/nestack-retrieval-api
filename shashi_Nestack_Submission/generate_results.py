#!/usr/bin/env python3
"""
generate_results.py - Evaluation & Result Generation Script
Executes benchmark test queries against the FastAPI retrieval endpoint (or directly via ChromaDB)
and outputs the results into results.json.
"""

import os
import sys
import json
import requests
from typing import List, Dict, Any

SERVER_URL = os.getenv("API_SERVER_URL", "http://localhost:8000/query")
OUTPUT_FILE = os.getenv("OUTPUT_FILE", "results.json")

# Benchmark evaluation queries representing key concepts in document.pdf
BENCHMARK_QUERIES = [
    "What embedding model is used for generating dense semantic vector representations?",
    "How does the text chunking strategy handle window size and boundary overlap?",
    "Which library is used to extract text and page numbers from the input PDF?",
    "What are the strict rules regarding LLMs and generative answers in the retrieval endpoint?",
    "How does ChromaDB store vectors and what distance metric is utilized?"
]


def query_api_server(query: str, top_k: int = 3) -> Dict[str, Any]:
    """Sends a query to the running FastAPI server endpoint."""
    response = requests.post(
        SERVER_URL,
        json={"query": query, "top_k": top_k},
        timeout=15
    )
    response.raise_for_status()
    return response.json()


def query_direct_chroma(query: str, top_k: int = 3) -> Dict[str, Any]:
    """
    Fallback method: directly queries ChromaDB if the HTTP server is not running.
    Useful for offline testing and standalone batch evaluation.
    """
    import chromadb
    from chromadb.utils import embedding_functions

    persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    collection_name = os.getenv("COLLECTION_NAME", "pdf_documents")
    model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    client = chromadb.PersistentClient(path=persist_dir)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)
    collection = client.get_collection(name=collection_name, embedding_function=ef)

    chroma_res = collection.query(
        query_texts=[query],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"]
    )

    documents = chroma_res.get("documents", [[]])[0]
    metadatas = chroma_res.get("metadatas", [[]])[0]
    distances = chroma_res.get("distances", [[]])[0]

    results = []
    for doc, meta, dist in zip(documents, metadatas, distances):
        score = max(0.0, 1.0 - float(dist)) if dist is not None else 0.0
        results.append({
            "chunk_text": doc,
            "page_number": int(meta.get("page_number", 1)),
            "score": round(score, 4)
        })

    return {
        "query": query,
        "results": results,
        "total_results": len(results)
    }


def main():
    print("=" * 65)
    print(" Nestack Assessment - Benchmark Evaluation & Result Generator")
    print("=" * 65)
    print(f"Target Server: {SERVER_URL}")
    print(f"Output File:   {OUTPUT_FILE}\n")

    use_api = True
    try:
        health_resp = requests.get(SERVER_URL.replace("/query", "/health"), timeout=2)
        if health_resp.status_code == 200:
            print("[INFO] Connected to live FastAPI server.")
        else:
            print("[WARN] Server responded with error, attempting direct query.")
            use_api = False
    except Exception:
        print("[NOTICE] API server not running locally. Falling back to direct ChromaDB engine.")
        use_api = False

    all_evaluations = []

    for idx, query in enumerate(BENCHMARK_QUERIES, 1):
        print(f"[{idx}/{len(BENCHMARK_QUERIES)}] Query: '{query}'")
        try:
            if use_api:
                data = query_api_server(query, top_k=2)
            else:
                data = query_direct_chroma(query, top_k=2)

            all_evaluations.append(data)
            print(f"  -> Retrieved {len(data.get('results', []))} chunks.")
            for r_idx, r in enumerate(data.get("results", []), 1):
                preview = r["chunk_text"][:90].replace("\n", " ")
                print(f"     #{r_idx} (Page {r['page_number']}, Score: {r['score']}): \"{preview}...\"")
        except Exception as e:
            print(f"  [ERROR] Failed to query: {e}", file=sys.stderr)

    output_payload = {
        "benchmark_summary": {
            "total_queries": len(BENCHMARK_QUERIES),
            "successful_evaluations": len(all_evaluations),
            "execution_mode": "FastAPI HTTP Endpoint" if use_api else "ChromaDB Direct Engine"
        },
        "evaluations": all_evaluations
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 65)
    print(f"SUCCESS: Results successfully written to '{OUTPUT_FILE}'.")
    print("=" * 65)


if __name__ == "__main__":
    main()

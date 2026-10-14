#!/usr/bin/env python3
"""
ingest.py - Part 1: Ingestion Pipeline
Extracts text and page numbers from a PDF, splits text into overlapping chunks,
generates embeddings using Sentence Transformers, and stores vectors + metadata in ChromaDB.
"""

import os
import sys
from typing import List, Dict, Any
import fitz  # PyMuPDF
import chromadb
from chromadb.utils import embedding_functions

# Configuration constants
PDF_PATH = os.getenv("PDF_FILE_PATH", "document.pdf")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "pdf_documents")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))      # Character count per chunk
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100")) # Overlapping character count


def extract_pages_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extracts text and page numbers from each page of the input PDF using PyMuPDF.
    Returns a list of dicts: [{"page_number": int, "text": str}]
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at path: {pdf_path}")

    doc = fitz.open(pdf_path)
    pages_data = []

    print(f"[1/4] Opened PDF '{pdf_path}' with {len(doc)} pages.")
    for page_idx in range(len(doc)):
        page = doc[page_idx]
        text = page.get_text("text").strip()
        page_number = page_idx + 1  # 1-indexed page numbering
        if text:
            pages_data.append({
                "page_number": page_number,
                "text": text
            })
            print(f"  - Page {page_number}: extracted {len(text)} characters.")
        else:
            print(f"  - Page {page_number}: no text found (empty or scanned image).")

    doc.close()
    return pages_data


def chunk_text_with_overlap(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Splits text into overlapping sliding-window chunks.
    Ensures context continuity across chunk boundaries.
    """
    if not text:
        return []

    chunks = []
    start = 0
    text_length = len(text)

    # Edge case: text smaller than chunk_size
    if text_length <= chunk_size:
        return [text]

    step = max(1, chunk_size - overlap)
    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= text_length:
            break
        start += step

    return chunks


def process_and_chunk_document(pages_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Processes each page's text into overlapping chunks, preserving metadata.
    Returns structured chunks with IDs, page numbers, and chunk text.
    """
    all_chunks = []
    chunk_counter = 0

    print(f"[2/4] Chunking pages (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    for page_info in pages_data:
        page_num = page_info["page_number"]
        page_text = page_info["text"]
        chunks = chunk_text_with_overlap(page_text, CHUNK_SIZE, CHUNK_OVERLAP)

        for idx_in_page, chunk_content in enumerate(chunks):
            chunk_counter += 1
            chunk_id = f"doc_p{page_num}_c{idx_in_page + 1}_{chunk_counter}"
            all_chunks.append({
                "id": chunk_id,
                "chunk_text": chunk_content,
                "page_number": page_num,
                "chunk_index": idx_in_page + 1
            })

    print(f"  -> Generated total of {len(all_chunks)} overlapping chunks.")
    return all_chunks


def get_chroma_collection():
    """
    Initializes ChromaDB persistent client and retrieves or creates the collection
    with cosine distance space and Sentence Transformers embedding function.
    """
    os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

    # Use ChromaDB's native SentenceTransformerEmbeddingFunction for consistency
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_NAME
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=sentence_transformer_ef,
        metadata={"hnsw:space": "cosine"}
    )
    return collection


def store_chunks_in_chromadb(chunks: List[Dict[str, Any]], collection) -> int:
    """
    Stores vectors, original chunk texts, and metadata into ChromaDB.
    Chroma computes embeddings automatically using the configured embedding function.
    """
    if not chunks:
        print("No chunks to store.")
        return 0

    print(f"[3/4 & 4/4] Generating embeddings ({EMBEDDING_MODEL_NAME}) and saving to ChromaDB...")

    batch_size = 64
    total_stored = 0

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        ids = [item["id"] for item in batch]
        documents = [item["chunk_text"] for item in batch]
        metadatas = [
            {
                "page_number": int(item["page_number"]),
                "chunk_index": int(item["chunk_index"])
            }
            for item in batch
        ]

        collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
        total_stored += len(batch)
        print(f"  - Ingested batch {i // batch_size + 1}: {len(batch)} chunks (total: {total_stored})")

    return total_stored


def main():
    print("=" * 60)
    print(" Nestack Assessment - Part 1: Document Ingestion Pipeline")
    print("=" * 60)

    pdf_target = sys.argv[1] if len(sys.argv) > 1 else PDF_PATH
    print(f"Target PDF: {pdf_target}")
    print(f"Persistent Vector DB: {CHROMA_PERSIST_DIR}")
    print(f"Collection Name: {COLLECTION_NAME}")

    try:
        pages = extract_pages_from_pdf(pdf_target)
        if not pages:
            print("Warning: No text extracted from PDF.")
            return

        chunks = process_and_chunk_document(pages)
        collection = get_chroma_collection()
        total_stored = store_chunks_in_chromadb(chunks, collection)

        print("\n" + "=" * 60)
        print(f"SUCCESS: Ingestion complete! {total_stored} chunks saved in ChromaDB.")
        print(f"Collection count: {collection.count()} vectors ready for retrieval.")
        print("=" * 60)
    except Exception as e:
        print(f"\nERROR during ingestion: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

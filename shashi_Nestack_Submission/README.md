# Nestack Assessment: PDF Ingestion & Semantic Retrieval System
**Candidate**: shashi
**Role**: Backend Development Engineer (Python / FastAPI / Vector DB)

A high-performance, local semantic search backend built with **Python 3.11**, **FastAPI**, **ChromaDB**, and **Sentence Transformers**.

> **Important Compliance Note**: This is a pure semantic search system, **not a chatbot**. The retrieval endpoint does not invoke any Large Language Model (LLM) to synthesize or hallucinate text answers. It calculates cosine similarity over dense vector embeddings and returns the verbatim text chunks with their source page numbers and similarity scores.

---

## 📁 Repository & Submission Directory Structure

```text
shashi_Nestack_Submission/
│
├── ingest.py              # Part 1: PDF extraction, sliding-window chunking, embedding & ChromaDB storage
├── server.py              # Part 2: FastAPI service exposing semantic search retrieval endpoints
├── generate_results.py    # Benchmark evaluation script running sample queries and logging results
├── requirements.txt       # Production dependencies (PyMuPDF, ChromaDB, Sentence-Transformers, FastAPI)
├── README.md              # Comprehensive setup, architecture, and API documentation
├── results.json           # Evaluation output with queries, verbatim chunks, page numbers, and scores
├── .gitignore             # Git ignore rules for virtual environments, chroma_db, and caches
├── .env.example           # Configurable environment variables template
│
└── document.pdf           # Sample multi-page PDF document used for ingestion & benchmarking
```

---

## 🛠️ Technology Stack & Architecture

| Technology | Purpose | Key Details |
| :--- | :--- | :--- |
| **Python 3.11** | Core Language | Modern type annotations, async runtime |
| **PyMuPDF (`fitz`)** | Text Extraction | Fast, robust page-by-page text parsing with explicit page tracking |
| **Sentence Transformers** | Vector Embeddings | `all-MiniLM-L6-v2` (384 dimensions) — runs locally on CPU with zero API costs |
| **ChromaDB** | Vector Database | Persistent HNSW index with cosine distance space (`hnsw:space: cosine`) |
| **FastAPI** | Web Framework | Lightweight, high-throughput REST API with automated OpenAPI docs |
| **Uvicorn** | ASGI Web Server | Production-ready asynchronous HTTP server |
| **Requests** | Testing & Client | Used in `generate_results.py` to benchmark query endpoints |

---

## 🚀 Step-by-Step Setup & Execution

### 1. Prerequisites & Virtual Environment

Ensure Python 3.10+ (recommended 3.11) is installed on your system.

```bash
# Enter submission directory
cd shashi_Nestack_Submission

# Create and activate virtual environment
python3 -m venv venv

# On Linux/macOS:
source venv/bin/activate

# On Windows (cmd/powershell):
# venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

### 3. Part 1 — Run Document Ingestion

The ingestion script extracts text from `document.pdf`, chunks it with an overlapping sliding window (500 chars window, 100 chars overlap), generates dense embeddings, and stores them in ChromaDB.

```bash
python ingest.py
```

**Expected Ingestion Output:**
```text
============================================================
 Nestack Assessment - Part 1: Document Ingestion Pipeline
============================================================
Target PDF: document.pdf
Persistent Vector DB: ./chroma_db
Collection Name: pdf_documents
[1/4] Opened PDF 'document.pdf' with 3 pages.
  - Page 1: extracted 782 characters.
  - Page 2: extracted 814 characters.
  - Page 3: extracted 836 characters.
[2/4] Chunking pages (chunk_size=500, overlap=100)...
  -> Generated total of 8 overlapping chunks.
[3/4 & 4/4] Generating embeddings (all-MiniLM-L6-v2) and saving to ChromaDB...
  - Ingested batch 1: 8 chunks (total: 8)

============================================================
SUCCESS: Ingestion complete! 8 chunks saved in ChromaDB.
Collection count: 8 vectors ready for retrieval.
============================================================
```

---

### 4. Part 2 — Start the Retrieval API Server

Launch the FastAPI server on port 8000:

```bash
uvicorn server:app --reload --host 0.0.0.0 --port 8000
```

- **Interactive API Documentation (Swagger)**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

---

### 5. Part 2 — Query the Retrieval API

You can query the API using `curl`, Python `requests`, or your browser.

#### Example using `curl` (POST):
```bash
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"query": "What embedding model is used for semantic search?", "top_k": 2}'
```

#### JSON Response Schema:
```json
{
  "query": "What embedding model is used for semantic search?",
  "results": [
    {
      "chunk_text": "2.1 Sentence Transformers & Dense Embeddings\nFor dense semantic vector representation, Nestack utilizes the all-MiniLM-L6-v2 sentence transformers model. This 384-dimensional embedding model maps semantic meanings into Euclidean space with rapid inference speeds, making it ideal for local CPU execution without paid APIs.",
      "page_number": 2,
      "score": 0.8914
    },
    {
      "chunk_text": "2.2 ChromaDB Persistence & Indexing\nChromaDB operates as the primary vector store. Ingested vectors are indexed using Hierarchical Navigable Small World (HNSW) graphs with cosine distance metrics. Persistent collections store chunk content, page numbers, and vector embeddings on local disk under the chroma_db directory.",
      "page_number": 2,
      "score": 0.6845
    }
  ],
  "total_results": 2
}
```

---

### 6. Run Benchmark Evaluation & Generate `results.json`

In a separate terminal:

```bash
python generate_results.py
```

This executes benchmark queries across all 3 pages of `document.pdf` and saves the outputs directly to `results.json`.

# Ares: Distributed Cloud-Native Vector Database

[![Architecture: Distributed](https://img.shields.io/badge/Architecture-Distributed%20Scatter--Gather-blue.svg)](#system-architecture)
[![Python: 3.12](https://img.shields.io/badge/Python-3.12-yellow.svg)](https://www.python.org/)
[![Framework: FastAPI](https://img.shields.io/badge/Framework-FastAPI-green.svg)](https://fastapi.tiangolo.com/)
[![Engine: Docker Compose](https://img.shields.io/badge/Container-Docker%20Compose-2496ED.svg)](https://www.docker.com/)
[![Embeddings: all-MiniLM-L6-v2](https://img.shields.io/badge/Embeddings-384--d%20MiniLM-orange.svg)](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

Ares is a distributed, horizontally scalable vector database built from first principles in Python and containerized via Docker. Engineered using a disaggregated compute-storage pattern, Ares decouples API ingestion and scatter-gather query orchestration from isolated, persistent compute nodes performing sub-millisecond approximate nearest neighbour (ANN) vector searches.

---

## System Architecture

Ares utilizes a 3-node cluster topology orchestrated via Docker Compose:

```text
                      +-------------------------+
                      |   Client / RAG Client   |
                      +------------+------------+
                                   |
                            HTTP (Port 9000)
                                   v
                 +-----------------------------------+
                 |          Ares Coordinator         |
                 |   (API Gateway & Sharding Router) |
                 +-----------------+-----------------+
                                   |
               +-------------------+-------------------+
               | Internal Docker Network               | Internal Docker Network
               | (Port 8000)                           | (Port 8000)
               v                                       v
 +---------------------------+           +---------------------------+
 |      ares_compute_01      |           |      ares_compute_02      |
 |   (Stateless Math Engine) |           |   (Stateless Math Engine) |
 +-------------+-------------+           +-------------+-------------+
               |                                       |
        Volume Mount                            Volume Mount
               v                                       v
     [ ./data/node_1/ ]                      [ ./data/node_2/ ]
  - corpus.json                           - corpus.json
  - embeddings.npy                        - embeddings.npy
````

### 1. Coordinator Node (`ares_coordinator` — Port 9000)

- **Ingress & Sharding Router:** Receives write payloads (`/ingest`) and shards incoming text records across active compute instances to balance storage footprints.
    
      
    
- **Scatter-Gather Engine:** For incoming queries (`/search`), concurrently broadcasts asynchronous HTTP requests (`httpx`) across all cluster nodes, pools score arrays, sorts candidates globally by cosine similarity, and reduces output to the Top-$K$ nearest neighbors.
    
      
    

### 2. Compute Nodes (`ares_compute_01`, `ares_compute_02` — Port 8000 Internal)

- **Isolated Vector Engines:** Pre-computes 384-dimensional dense vectors using the `all-MiniLM-L6-v2` transformer model.
    
      
    
- **Vectorized Linear Algebra:** Implements vectorized cosine similarity calculations using `numpy` dot products and Euclidean norms:
    
      
    
    $$\text{Similarity}(A, B) = \frac{A \cdot B}{\Vert{}A\Vert{}_2 \Vert{}B\Vert{}_2}$$
    
- **Zero-Amnesia Persistence:** Persists local document lists and NumPy matrix arrays directly to host disk volumes (`./data/node_*`), ensuring node restarts recover state without data loss.
    
      
    

### 3. Resilient RAG Integration (`rag_pipeline.py`)

- End-to-end Retrieval-Augmented Generation client querying the Coordinator gateway.
    
      
    
- Enforces strict, anti-hallucination prompting context against Google Gemini (`gemini-2.5-flash-lite`).
    
      
    
- Hardened with exponential backoff and decorrelated jitter to guarantee network resilience against upstream provider rate limits.
    
      
    

## Tech Stack

- **Language:** Python 3.12
    
      
    
- **API Framework:** FastAPI, Uvicorn, Pydantic
    
      
    
- **Math & Embeddings:** NumPy, PyTorch, Hugging Face `sentence-transformers`
    
      
    
- **Cluster Networking:** HTTPX (Asynchronous Client)
    
      
    
- **Containerization:** Docker Desktop, Docker Compose
    
      
    
- **Generative Engine:** Google GenAI SDK
    
      
    

## Directory Structure

```
Project_Ares/
├── data/
│   ├── node_1/              # compute_01 persistent index & embeddings
│   └── node_2/              # compute_02 persistent index & embeddings
├── .env                     # Local API keys (HF_TOKEN, GEMINI_API_KEY)
├── .gitignore               # Secrets and data volume exclusions
├── compute_node.py          # Isolated Vector Engine & Storage Worker
├── coordinator.py           # API Gateway, Sharding Router & Aggregator
├── Dockerfile               # Production container image definition
├── docker-compose.yml       # 3-Node cluster orchestration manifest
├── rag_pipeline.py          # Resilient client & anti-hallucination pipeline
├── requirements.txt         # Pinned production dependencies
└── README.md
```

## API Reference

### 1. Ingest Documents (Distributed Sharding)

Distributes a collection of text documents across the active compute node cluster.

  

- **Endpoint:** `POST http://localhost:9000/ingest`
    
      
    
- **Header:** `Content-Type: application/json`
    
      
    

**Request Payload:**

  

```
{
  "documents": [
    "Quantum computers use qubits to perform calculations.",
    "The Great Wall of China is visible from low Earth orbit.",
    "Photosynthesis converts light energy into chemical energy."
  ]
}
```

**Response (`200 OK`):**



```
{
  "message": "Distributed ingestion complete.",
  "details": [
    {
      "doc": "Quantum computers use qubits to perform calculations.",
      "node": "http://compute_1:8000",
      "status": 200
    },
    {
      "doc": "The Great Wall of China is visible from low Earth orbit.",
      "node": "http://compute_2:8000",
      "status": 200
    },
    {
      "doc": "Photosynthesis converts light energy into chemical energy.",
      "node": "http://compute_1:8000",
      "status": 200
    }
  ]
}
```

### 2. Distributed Search (Scatter-Gather)

Scatters query embeddings across all nodes, collects local similarities, and returns the globally ranked Top-$K$ records.

  

- **Endpoint:** `POST http://localhost:9000/search`
    
      
    
- **Header:** `Content-Type: application/json`
    
      
    

**Request Payload:**

  

```
{
  "text": "How do plants make food?",
  "top_k": 2
}
```

**Response (`200 OK`):**

  

```
{
  "query": "How do plants make food?",
  "results": [
    {
      "rank": 1,
      "score": 0.6841,
      "document": "Photosynthesis converts light energy into chemical energy."
    },
    {
      "rank": 2,
      "score": 0.1219,
      "document": "The Great Wall of China is visible from low Earth orbit."
    }
  ]
}
```

## Quickstart & Deployment

### Prerequisites

- Docker Desktop installed and running with WSL2 integration enabled (if running on Windows).
    
      
    
- Python 3.12+ virtual environment.
    
      
    

### 1. Configure Environment Secrets

Create a `.env` file in the root directory:


```
HF_TOKEN=your_huggingface_token_here
GEMINI_API_KEY=your_gemini_api_key_here
```

### 2. Build and Launch the Cluster

```
# Build base images and launch all three nodes in background
docker compose up --build -d

# Verify node health and port bindings
docker compose ps
```

### 3. Verify Local Persistence

Check the host filesystem to verify disk volume mounts:

  
```
ls -la ./data/node_1
ls -la ./data/node_2
```

### 4. Execute the RAG Pipeline

Run the fault-tolerant client to issue queries through the distributed cluster into Gemini:



```
python rag_pipeline.py
```

## Project Roadmap

- [x] **Phase 1: Local Containerization & Sharding:** Disaggregated 3-node cluster with scatter-gather routing and volume persistence.
    
      
    
- [ ] **Phase 2: Kubernetes (K8s) Orchestration:** Translating manifests to K8s Deployments, StatefulSets for storage, and HPA autoscaling.
    
      
    
- [ ] **Phase 3: CI/CD Pipeline Automation:** GitHub Actions workflow with automated load testing and cosine math invariant checks.
    
      
    
- [ ] **Phase 4: Cloud Provisioning:** Production multi-node deployment on AWS EKS with S3-backed durable persistence.

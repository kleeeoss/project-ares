import os
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

SNAPSHOT_DIR = os.getenv("PERSISTENCE_DIR", "/data")
SNAPSHOT_FILE = os.path.join(SNAPSHOT_DIR, "shard_index.npz")

documents = []
embeddings = None
model = None


def persist_snapshot():
    """Atomically writes memory index to disk using a temporary swap file."""
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    temp_file = f"{SNAPSHOT_FILE}.tmp.npz"

    # Save both text records and normalized float32 embedding matrix
    np.savez_compressed(
        temp_file,
        documents=np.array(documents, dtype=object),
        embeddings=embeddings if embeddings is not None else np.empty((0, 384), dtype=np.float32)
    )
    os.replace(temp_file, SNAPSHOT_FILE)


def load_snapshot():
    """Restores documents and embeddings from disk on pod startup."""
    global documents, embeddings
    if os.path.exists(SNAPSHOT_FILE):
        data = np.load(SNAPSHOT_FILE, allow_pickle=True)
        documents = data["documents"].tolist()
        embeddings = data["embeddings"]
        print(f"📦 [Hydration] Restored {len(documents)} vectors from {SNAPSHOT_FILE}", flush=True)
    else:
        print(f"ℹ️ [Hydration] No snapshot found at {SNAPSHOT_FILE}. Starting cold shard.", flush=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model = SentenceTransformer("all-MiniLM-L6-v2")
    load_snapshot()
    yield


app = FastAPI(title="Ares Compute Node", lifespan=lifespan)


class IngestRequest(BaseModel):
    documents: list[str]


class SearchRequest(BaseModel):
    text: str
    top_k: int = 3


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "documents_indexed": len(documents),
        "snapshot_exists": os.path.exists(SNAPSHOT_FILE)
    }


@app.post("/ingest")
def ingest(payload: IngestRequest):
    global documents, embeddings
    if not payload.documents:
        raise HTTPException(status_code=400, detail="Empty document payload.")

    new_vectors = model.encode(payload.documents, normalize_embeddings=True)

    if embeddings is None or len(embeddings) == 0:
        embeddings = new_vectors
    else:
        embeddings = np.vstack([embeddings, new_vectors])

    documents.extend(payload.documents)
    persist_snapshot()

    return {"status": "ok", "indexed_count": len(payload.documents), "total_count": len(documents)}


@app.post("/search")
def search(payload: SearchRequest):
    if embeddings is None or len(documents) == 0:
        return {"results": []}

    query_vec = model.encode(payload.text, normalize_embeddings=True)
    scores = np.dot(embeddings, query_vec)

    top_indices = np.argsort(scores)[::-1][:payload.top_k]
    results = [
        {"document": documents[idx], "score": float(scores[idx])}
        for idx in top_indices
    ]
    return {"results": results}
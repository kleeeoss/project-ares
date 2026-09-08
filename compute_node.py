import numpy as np
import json
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import warnings
import uvicorn
from dotenv import load_dotenv

warnings.filterwarnings("ignore")

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

app = FastAPI(title="Ares Compute Node", description="Persistent Vector Search Node")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Define where the physical files will live
DATA_DIR = "./data"
os.makedirs(DATA_DIR, exist_ok=True)
CORPUS_FILE = f"{DATA_DIR}/corpus.json"
EMBEDDINGS_FILE = f"{DATA_DIR}/embeddings.npy"


# --- STORAGE LOGIC ---
def load_index():
    if os.path.exists(CORPUS_FILE) and os.path.exists(EMBEDDINGS_FILE):
        print("Loading existing Vector Index from disk...")
        with open(CORPUS_FILE, 'r') as f:
            corpus = json.load(f)
        corpus_embeddings = np.load(EMBEDDINGS_FILE)
    else:
        print("⚠️ No index found. Creating default index...")
        corpus = [
            "A fast, dark-colored canine leaps above a sleepy hound.",
            "Cloud computing allows scalable infrastructure deployment.",
            "The Apollo 11 mission landed humans on the moon in 1969."
        ]
        corpus_embeddings = model.encode(corpus)
        save_index(corpus, corpus_embeddings)

    norms = np.linalg.norm(corpus_embeddings, axis=1)
    return corpus, corpus_embeddings, norms


def save_index(corpus, embeddings):
    print("💾 Saving updated Vector Index to disk...")
    with open(CORPUS_FILE, 'w') as f:
        json.dump(corpus, f)
    np.save(EMBEDDINGS_FILE, embeddings)


# Boot Sequence
print("Booting Ares Compute Node...")
corpus, corpus_embeddings, corpus_norms = load_index()


# --- NETWORK PAYLOADS ---
class SearchQuery(BaseModel):
    text: str
    top_k: int = 3


class IngestPayload(BaseModel):
    documents: list[str]


# --- ENDPOINTS ---
@app.post("/ingest")
async def ingest_data(payload: IngestPayload):
    global corpus, corpus_embeddings, corpus_norms

    if not payload.documents:
        raise HTTPException(status_code=400, detail="No documents provided.")

    print(f" Ingesting {len(payload.documents)} new documents...")
    new_embeddings = model.encode(payload.documents)

    # Update memory
    corpus_embeddings = np.vstack([corpus_embeddings, new_embeddings])
    corpus.extend(payload.documents)
    corpus_norms = np.linalg.norm(corpus_embeddings, axis=1)

    # Save to disk!
    save_index(corpus, corpus_embeddings)

    return {"message": "Ingestion successful.", "total_index_size": len(corpus)}


@app.post("/search")
async def search(query: SearchQuery):
    query_embedding = model.encode(query.text)
    query_norm = np.linalg.norm(query_embedding)

    if query_norm == 0:
        return {"results": []}

    dot_products = np.dot(corpus_embeddings, query_embedding)
    similarities = dot_products / (query_norm * corpus_norms)
    top_k_indices = np.argsort(similarities)[::-1][:query.top_k]

    results = [{"rank": i + 1, "score": float(similarities[idx]), "document": corpus[idx]} for i, idx in
               enumerate(top_k_indices)]
    return {"query": query.text, "results": results}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
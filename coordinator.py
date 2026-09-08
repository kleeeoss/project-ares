import asyncio
import httpx
import random
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="Ares Coordinator Node", description="Distributed Scatter-Gather Router")

COMPUTE_NODES = [
    "http://compute_1:8000",
    "http://compute_2:8000"
]


class SearchQuery(BaseModel):
    text: str
    top_k: int = 3


class IngestPayload(BaseModel):
    documents: list[str]


# 🆕 THE SHARDING ROUTER
@app.post("/ingest")
async def distributed_ingest(payload: IngestPayload):
    print(f"📦 [Coordinator] Routing {len(payload.documents)} documents to storage cluster...")

    async with httpx.AsyncClient() as client:
        results = []
        for doc in payload.documents:
            # SHARDING ALGORITHM: Randomly select a node to store this specific document
            target_node = random.choice(COMPUTE_NODES)
            print(f"🔀 Sharding document to: {target_node}")

            # Forward the data to the internal Docker network
            response = await client.post(f"{target_node}/ingest", json={"documents": [doc]})
            results.append({"doc": doc, "node": target_node, "status": response.status_code})

    return {"message": "Distributed ingestion complete.", "details": results}


# THE SCATTER-GATHER SEARCH
@app.post("/search")
async def distributed_search(query: SearchQuery):
    print(f"🌐 [Coordinator] Received query: '{query.text}'. Scattering to cluster...")

    async with httpx.AsyncClient() as client:
        tasks = []
        for node in COMPUTE_NODES:
            tasks.append(client.post(f"{node}/search", json={"text": query.text, "top_k": query.top_k}))

        try:
            responses = await asyncio.gather(*tasks)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Cluster communication error: {str(e)}")

    all_results = []
    for response in responses:
        if response.status_code == 200:
            data = response.json()
            all_results.extend(data.get("results", []))

    # REDUCE: Sort mathematically by Cosine Similarity score
    sorted_results = sorted(all_results, key=lambda x: x['score'], reverse=True)
    final_top_k = sorted_results[:query.top_k]

    for i, result in enumerate(final_top_k):
        result['rank'] = i + 1

    print(f"✅ [Coordinator] Gathered and sorted {len(all_results)} total vectors. Returning Top {len(final_top_k)}.")
    return {"query": query.text, "results": final_top_k}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000)
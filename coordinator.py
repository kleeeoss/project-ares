import asyncio
import httpx
import random
import socket
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import os
import json

app = FastAPI(title="Ares Coordinator Node", description="Distributed Scatter-Gather Router")

COMPUTE_NODES = ["http://compute_1:8000", "http://compute_2:8000"]
nodes_env = os.getenv("COMPUTE_NODES")

if nodes_env:
    COMPUTE_NODES = json.loads(nodes_env)

print(f"📡 [Coordinator] Active Compute Cluster Targets: {COMPUTE_NODES}")

_DNS_CACHE = {"nodes": [], "expiry": 0.0}
DNS_TTL_SECONDS = 5.0


async def get_active_compute_nodes() -> list[str]:
    """
    Non-blocking CoreDNS resolution with TTL cache.
    Resolves Headless Kubernetes Service endpoints without blocking the asyncio loop.
    Falls back to static COMPUTE_NODES if headless lookup is not configured or fails.
    """
    now = time.time()
    if _DNS_CACHE["nodes"] and now < _DNS_CACHE["expiry"]:
        return _DNS_CACHE["nodes"]

    headless_host = os.getenv("COMPUTE_HEADLESS_HOST")
    if headless_host:
        try:
            loop = asyncio.get_running_loop()
            _, _, ip_list = await loop.run_in_executor(None, socket.gethostbyname_ex, headless_host)
            if ip_list:
                nodes = [f"http://{ip}:8000" for ip in ip_list]
                _DNS_CACHE["nodes"] = nodes
                _DNS_CACHE["expiry"] = now + DNS_TTL_SECONDS
                return nodes
        except Exception as e:
            print(f"⚠️ [DNS] Headless lookup failed on '{headless_host}': {e}. Falling back to default nodes.")

    return COMPUTE_NODES


class SearchQuery(BaseModel):
    text: str
    top_k: int = 3


class IngestPayload(BaseModel):
    documents: list[str]


# 🆕 THE BATCHED SHARDING ROUTER
@app.post("/ingest")
async def distributed_ingest(payload: IngestPayload):
    if not payload.documents:
        raise HTTPException(status_code=400, detail="Empty document payload.")

    nodes = await get_active_compute_nodes()
    if not nodes:
        raise HTTPException(status_code=503, detail="No active compute nodes available")

    print(f"📦 [Coordinator] Routing {len(payload.documents)} documents across {len(nodes)} shards...")

    # Round-robin document bucketing across active compute nodes
    node_buckets: dict[str, list[str]] = {node: [] for node in nodes}
    for idx, doc in enumerate(payload.documents):
        target_node = nodes[idx % len(nodes)]
        node_buckets[target_node].append(doc)

    async with httpx.AsyncClient(timeout=30.0) as client:
        tasks = []
        active_targets = []
        for node, docs in node_buckets.items():
            if docs:
                tasks.append(client.post(f"{node}/ingest", json={"documents": docs}))
                active_targets.append((node, docs))

        responses = await asyncio.gather(*tasks, return_exceptions=True)

    results = []
    failed_batches = 0
    for (node, docs), resp in zip(active_targets, responses):
        if isinstance(resp, Exception) or resp.status_code >= 400:
            failed_batches += 1
            err_msg = str(resp) if isinstance(resp, Exception) else f"HTTP {resp.status_code}: {resp.text}"
            status_val = 500 if isinstance(resp, Exception) else resp.status_code
            results.append({"node": node, "count": len(docs), "status": status_val, "error": err_msg})
            print(f"⚠️ [Coordinator] Ingest failed on '{node}': {err_msg}")
        else:
            results.append({"node": node, "count": len(docs), "status": resp.status_code})

    if failed_batches == len(active_targets):
        raise HTTPException(status_code=503, detail=f"All compute shards failed during ingestion: {results}")

    return {
        "message": "Distributed ingestion complete.",
        "documents_routed": len(payload.documents),
        "details": results
    }


# THE RESILIENT SCATTER-GATHER SEARCH
@app.post("/search")
async def distributed_search(query: SearchQuery):
    print(f"🌐 [Coordinator] Received query: '{query.text}'. Scattering to cluster...")
    nodes = await get_active_compute_nodes()
    if not nodes:
        raise HTTPException(status_code=503, detail="No active compute nodes available")

    async with httpx.AsyncClient(timeout=10.0) as client:
        tasks = [
            client.post(f"{node}/search", json={"text": query.text, "top_k": query.top_k})
            for node in nodes
        ]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

    all_results = []
    successful_nodes = 0
    failed_nodes = 0
    node_errors = []

    for node, resp in zip(nodes, responses):
        if isinstance(resp, Exception):
            failed_nodes += 1
            node_errors.append({"node": node, "error": str(resp)})
            print(f"⚠️ [Coordinator] Shard communication error on '{node}': {resp}")
        elif resp.status_code == 200:
            successful_nodes += 1
            try:
                data = resp.json()
                all_results.extend(data.get("results", []))
            except Exception as e:
                failed_nodes += 1
                node_errors.append({"node": node, "error": f"Invalid JSON response: {e}"})
        else:
            failed_nodes += 1
            node_errors.append({"node": node, "error": f"HTTP {resp.status_code}: {resp.text}"})

    if successful_nodes == 0:
        raise HTTPException(
            status_code=503,
            detail=f"All compute shards failed. Errors: {node_errors}"
        )

    # REDUCE: Sort mathematically by Cosine Similarity score descending
    sorted_results = sorted(all_results, key=lambda x: x.get("score", 0.0), reverse=True)
    final_top_k = sorted_results[:query.top_k]

    for i, result in enumerate(final_top_k):
        result["rank"] = i + 1

    print(f"✅ [Coordinator] Gathered {len(all_results)} vectors from {successful_nodes}/{len(nodes)} shards. Returning Top {len(final_top_k)}.")
    return {
        "query": query.text,
        "results": final_top_k,
        "cluster_health": {
            "total_shards": len(nodes),
            "successful_shards": successful_nodes,
            "failed_shards": failed_nodes,
            "partial_results": failed_nodes > 0
        }
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000)

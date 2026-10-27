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
from typing import Optional, List
from google import genai
from google.genai import types
from genai_client import get_vertex_client

app = FastAPI(title="Ares Coordinator Node", description="Distributed Scatter-Gather Router with HyDE")

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
    use_hyde: bool = False


class IngestPayload(BaseModel):
    documents: list[str]


async def expand_query_hyde(
    query_text: str,
    client: Optional[genai.Client] = None,
    timeout: float = 5.0
) -> str:
    """
    Expands a search query using Hypothetical Document Embeddings (HyDE) via Vertex AI.
    Falls back gracefully to the raw query text on any error or timeout.
    Isolated strictly to the coordinator ingress layer (compute shards remain untouched).
    """
    if not query_text or not query_text.strip():
        return query_text

    try:
        genai_client = client or get_vertex_client()
        if genai_client is None:
            return query_text

        prompt = (
            f"Please write a short, informative hypothetical passage answering the following search query. "
            f"Write only the passage without preamble or meta-commentary:\n\n{query_text}"
        )
        config = types.GenerateContentConfig(
            temperature=0.2,
            thinking_config=types.ThinkingConfig(thinking_budget=0)
        )
        model_id = os.getenv("GEMINI_MODEL_ID", "gemini-3.5-flash-lite")

        resp = await asyncio.wait_for(
            genai_client.aio.models.generate_content(
                model=model_id,
                contents=prompt,
                config=config
            ),
            timeout=timeout
        )
        text = resp.text.strip() if resp and resp.text else ""
        return text if text else query_text
    except Exception as e:
        print(f"⚠️ [HyDE] Query expansion failed or timed out: {e}. Falling back to raw query.")
        return query_text


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


# THE RESILIENT SCATTER-GATHER SEARCH (WITH COORDINATOR HYDE)
@app.post("/search")
async def distributed_search(query: SearchQuery):
    print(f"🌐 [Coordinator] Received query: '{query.text}'. Scattering to cluster...")
    nodes = await get_active_compute_nodes()
    if not nodes:
        raise HTTPException(status_code=503, detail="No active compute nodes available")

    search_text = query.text
    hyde_applied = False
    if query.use_hyde:
        expanded = await expand_query_hyde(query.text)
        if expanded and expanded != query.text:
            print(f"✨ [HyDE] Query expanded: '{query.text}' -> '{expanded[:60]}...'")
            search_text = expanded
            hyde_applied = True

    async with httpx.AsyncClient(timeout=10.0) as client:
        tasks = [
            client.post(f"{node}/search", json={"text": search_text, "top_k": query.top_k})
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
        "search_text": search_text,
        "hyde_applied": hyde_applied,
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

import asyncio
import time
import httpx
import numpy as np

COORDINATOR_URL = "http://localhost:9000/search"
TOTAL_REQUESTS = 50
MAX_CONCURRENCY = 5
PAYLOAD = {
    "text": "Kubernetes orchestration for distributed neural network inference",
    "top_k": 3
}

async def send_query(client: httpx.AsyncClient, semaphore: asyncio.Semaphore, latencies: list):
    async with semaphore:
        start = time.perf_counter()
        try:
            response = await client.post(COORDINATOR_URL, json=PAYLOAD, timeout=10.0)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            if response.status_code == 200:
                latencies.append(elapsed_ms)
            else:
                print(f"⚠️ Query failed with status: {response.status_code}")
        except Exception as e:
            print(f"❌ Connection error: {e}")

async def warmup(client: httpx.AsyncClient):
    print("🔥 Priming connection pools and PyTorch runtime...")
    for _ in range(3):
        try:
            await client.post(COORDINATOR_URL, json=PAYLOAD, timeout=10.0)
        except Exception:
            pass

async def run_benchmark():
    latencies = []
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
    limits = httpx.Limits(max_keepalive_connections=20, max_connections=20)

    async with httpx.AsyncClient(limits=limits) as client:
        # 1. Warm-up to eliminate cold-start noise from percentiles
        await warmup(client)

        # 2. Measure steady-state performance
        print(f"🚀 Firing {TOTAL_REQUESTS} steady-state requests ({MAX_CONCURRENCY} concurrent) against {COORDINATOR_URL}...")
        tasks = [send_query(client, semaphore, latencies) for _ in range(TOTAL_REQUESTS)]
        await asyncio.gather(*tasks)

    if not latencies:
        print("❌ All requests failed. Ensure port-forwarding on 9000 is active.")
        return

    latencies = np.array(latencies)
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)

    print("\n--- Ares Benchmark Latency Report ---")
    print(f"Successful Queries : {len(latencies)} / {TOTAL_REQUESTS}")
    print(f"Min Latency        : {np.min(latencies):.2f} ms")
    print(f"p50 (Median)       : {p50:.2f} ms")
    print(f"p95 Latency        : {p95:.2f} ms")
    print(f"p99 Latency        : {p99:.2f} ms")
    print(f"Max Latency        : {np.max(latencies):.2f} ms")
    print("-------------------------------------")

    assert p95 < 1000.0, f"p95 latency {p95:.2f}ms exceeded 1000ms threshold!"
    print("✅ Target verified: sub-second steady-state response times achieved.")

if __name__ == "__main__":
    asyncio.run(run_benchmark())
import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import httpx
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
import coordinator


# Mock compute shard simulating disk snapshot write + batch vectorization
class SimulatedComputeShard:
    def __init__(self, name: str):
        self.name = name
        self.doc_count = 0

    async def handle_ingest(self, request):
        data = await request.json()
        docs = data.get('documents', [])
        # Simulated cost: base 5ms overhead + 1ms per doc + 10ms snapshot persist
        await asyncio.sleep(0.015 + (0.001 * len(docs)))
        self.doc_count += len(docs)
        return JSONResponse({'status': 'ok', 'indexed': len(docs), 'total': self.doc_count})


async def run_benchmark(num_docs: int = 50):
    print('==================================================')
    print(f'ARES DISTRIBUTED INGESTION BENCHMARK ({num_docs} Documents)')
    print('==================================================')

    test_docs = [
        f'Document {i}: Distributed systems architectures, vector databases, and consensus mechanisms.'
        for i in range(num_docs)
    ]

    # Setup 2 simulated compute shards with ASGI mock transports
    shard_1 = SimulatedComputeShard('compute_1')
    shard_2 = SimulatedComputeShard('compute_2')

    app1 = Starlette(routes=[Route('/ingest', shard_1.handle_ingest, methods=['POST'])])
    app2 = Starlette(routes=[Route('/ingest', shard_2.handle_ingest, methods=['POST'])])

    transport1 = httpx.ASGITransport(app=app1)
    transport2 = httpx.ASGITransport(app=app2)

    def custom_client_factory():
        mounts = {
            'http://compute_1:8000': transport1,
            'http://compute_2:8000': transport2,
        }
        return httpx.AsyncClient(mounts=mounts, timeout=30.0)

    # ----------------------------------------------------
    # 1. SEQUENTIAL (DOCUMENT-BY-DOCUMENT) INGESTION
    # ----------------------------------------------------
    print('[1] Running Phase 1: Sequential Single-Doc Ingestion...')
    t0 = time.perf_counter()
    async with custom_client_factory() as client:
        nodes = coordinator.COMPUTE_NODES
        for idx, doc in enumerate(test_docs):
            target = nodes[idx % len(nodes)]
            resp = await client.post(f'{target}/ingest', json={'documents': [doc]})
            assert resp.status_code == 200
    t_seq = time.perf_counter() - t0
    throughput_seq = num_docs / t_seq
    print(f'   - Total Time: {t_seq:.3f} s')
    print(f'   - Throughput: {throughput_seq:.2f} docs/sec')

    # ----------------------------------------------------
    # 2. CONCURRENT BATCHED INGESTION
    # ----------------------------------------------------
    print('[2] Running Phase 2: Concurrent Batched Ingestion...')
    t0 = time.perf_counter()
    async with custom_client_factory() as client:
        nodes = coordinator.COMPUTE_NODES
        node_buckets = {node: [] for node in nodes}
        for idx, doc in enumerate(test_docs):
            target = nodes[idx % len(nodes)]
            node_buckets[target].append(doc)

        tasks = [
            client.post(f'{node}/ingest', json={'documents': docs})
            for node, docs in node_buckets.items()
            if docs
        ]
        responses = await asyncio.gather(*tasks)
        for resp in responses:
            assert resp.status_code == 200
    t_batch = time.perf_counter() - t0
    throughput_batch = num_docs / t_batch
    speedup = t_seq / t_batch if t_batch > 0 else 0

    print(f'   - Total Time: {t_batch:.3f} s')
    print(f'   - Throughput: {throughput_batch:.2f} docs/sec')

    # ----------------------------------------------------
    # SUMMARY REPORT
    # ----------------------------------------------------
    print('==================================================')
    print('EMPIRICAL BENCHMARK RESULTS')
    print('==================================================')
    print(f'   - Sequential Single-Doc : {throughput_seq:8.2f} docs/sec ({t_seq:.3f}s)')
    print(f'   - Concurrent Batched    : {throughput_batch:8.2f} docs/sec ({t_batch:.3f}s)')
    print(f'   - Empirical Speedup     : {speedup:8.2f}x')
    print('==================================================')
    return {
        'sequential_docs_per_sec': throughput_seq,
        'batched_docs_per_sec': throughput_batch,
        'speedup': speedup
    }


if __name__ == '__main__':
    asyncio.run(run_benchmark())

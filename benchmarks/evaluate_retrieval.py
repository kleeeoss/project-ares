import argparse
import asyncio
import json
import os
import sys
import time
from typing import Dict, List, Any

# Ensure Ares root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from sentence_transformers import SentenceTransformer
from coordinator import expand_query_hyde
from genai_client import get_vertex_client


def calculate_rank(scores: np.ndarray, target_idx: int) -> int:
    """Returns 1-indexed rank of target_idx based on descending score order."""
    sorted_indices = np.argsort(-scores).tolist()
    return sorted_indices.index(target_idx) + 1


def calculate_recall_at_k(rank: int, k: int) -> float:
    """Returns 1.0 if rank is within top-k, 0.0 otherwise."""
    return 1.0 if rank <= k else 0.0


def calculate_mrr(rank: int) -> float:
    """Returns reciprocal rank: 1 / rank."""
    return 1.0 / float(rank) if rank > 0 else 0.0


def calculate_rank_delta(base_rank: int, hyde_rank: int) -> int:
    """
    Returns positive value when HyDE improves rank (base_rank - hyde_rank).
    Positive: HyDE improved rank.
    Zero: No change.
    Negative: HyDE degraded rank.
    """
    return base_rank - hyde_rank


async def run_evaluation(mode: str = "offline", data_path: str = None) -> Dict[str, Any]:
    if data_path is None:
        data_path = os.path.join(os.path.dirname(__file__), "data", "ir_test_suite.json")

    with open(data_path, "r") as f:
        suite = json.load(f)

    corpus = suite["corpus"]
    queries = suite["queries"]
    doc_ids = [d["id"] for d in corpus]
    doc_texts = [d["text"] for d in corpus]

    print("================================================================================")
    header_mode = "[OFFLINE SYNTHETIC EXPANSIONS]" if mode == "offline" else "[LIVE VERTEX AI - gemini-3.5-flash-lite]"
    print(f"🔬 ARES INFORMATION RETRIEVAL BENCHMARK: BASELINE vs HyDE {header_mode}")
    print(f"📊 Corpus Size: {len(corpus)} documents | Query Count: {len(queries)} queries (5 Helps / 5 Neutral / 5 Hurts)")
    print("================================================================================")

    # 1. Precompute corpus embeddings once using SentenceTransformer with L2 unit normalization
    t_start = time.perf_counter()
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    corpus_embeddings = embedder.encode(doc_texts, normalize_embeddings=True)
    t_embed_corpus = time.perf_counter() - t_start
    print(f"⚡ Encoded 30-document corpus into 384-d normalized matrix in {t_embed_corpus:.3f}s\n")

    vertex_client = None
    if mode == "live":
        vertex_client = get_vertex_client(strict_production=True)
        print("🔗 Connected to Vertex AI via Google ADC (location: global, model: gemini-3.5-flash-lite)")

    detailed_results = []
    latencies = []

    for q_item in queries:
        q_id = q_item["id"]
        category = q_item["category"]
        query_text = q_item["query"]
        target_id = q_item["target_doc_id"]
        target_idx = doc_ids.index(target_id)

        # Baseline Retrieval: raw query -> flat BLAS dot product
        base_emb = embedder.encode(query_text, normalize_embeddings=True)
        base_scores = np.dot(corpus_embeddings, base_emb)
        base_rank = calculate_rank(base_scores, target_idx)

        # HyDE Expansion
        t_hyde_start = time.perf_counter()
        if mode == "live":
            expanded_text = await expand_query_hyde(query_text, client=vertex_client)
        else:
            expanded_text = q_item.get("synthetic_expansion", query_text)
        hyde_latency = time.perf_counter() - t_hyde_start
        latencies.append(hyde_latency)

        # HyDE Retrieval: expanded query -> flat BLAS dot product
        hyde_emb = embedder.encode(expanded_text, normalize_embeddings=True)
        hyde_scores = np.dot(corpus_embeddings, hyde_emb)
        hyde_rank = calculate_rank(hyde_scores, target_idx)

        delta = calculate_rank_delta(base_rank, hyde_rank)
        detailed_results.append({
            "id": q_id,
            "category": category,
            "query": query_text,
            "target_id": target_id,
            "base_rank": base_rank,
            "hyde_rank": hyde_rank,
            "delta": delta,
            "latency_s": hyde_latency
        })

    # Print Per-Query Results Table
    print(f"{'ID':4s} | {'CATEGORY':8s} | {'TARGET':7s} | {'BASE RANK':9s} | {'HyDE RANK':9s} | {'DELTA':7s} | {'HYDE TIME':9s}")
    print("-" * 75)
    for r in detailed_results:
        delta_str = f"+{r['delta']}" if r['delta'] > 0 else str(r['delta'])
        lat_str = f"{r['latency_s']*1000:6.1f} ms" if mode == "live" else f"{r['latency_s']*1000:6.2f} ms"
        print(f"{r['id']:4s} | {r['category'].upper():8s} | {r['target_id']:7s} | Rank {r['base_rank']:2d}   | Rank {r['hyde_rank']:2d}   | {delta_str:7s} | {lat_str:9s}")
    print("-" * 75)

    # Compute Aggregate Metrics
    n = len(detailed_results)
    base_rec5 = sum(calculate_recall_at_k(r["base_rank"], 5) for r in detailed_results) / n
    hyde_rec5 = sum(calculate_recall_at_k(r["hyde_rank"], 5) for r in detailed_results) / n

    base_rec10 = sum(calculate_recall_at_k(r["base_rank"], 10) for r in detailed_results) / n
    hyde_rec10 = sum(calculate_recall_at_k(r["hyde_rank"], 10) for r in detailed_results) / n

    base_mrr = sum(calculate_mrr(r["base_rank"]) for r in detailed_results) / n
    hyde_mrr = sum(calculate_mrr(r["hyde_rank"]) for r in detailed_results) / n

    # Category Breakdowns
    cat_metrics = {}
    for cat in ["helps", "neutral", "hurts"]:
        sub = [r for r in detailed_results if r["category"] == cat]
        sub_n = len(sub)
        cat_metrics[cat] = {
            "count": sub_n,
            "base_rec5": sum(calculate_recall_at_k(r["base_rank"], 5) for r in sub) / sub_n,
            "hyde_rec5": sum(calculate_recall_at_k(r["hyde_rank"], 5) for r in sub) / sub_n,
            "base_mrr": sum(calculate_mrr(r["base_rank"]) for r in sub) / sub_n,
            "hyde_mrr": sum(calculate_mrr(r["hyde_rank"]) for r in sub) / sub_n,
            "mean_delta": sum(r["delta"] for r in sub) / sub_n
        }

    print("\n================================================================================")
    print("📈 AGGREGATE INFORMATION RETRIEVAL PERFORMANCE SUMMARY")
    print("================================================================================")
    print(f"{'Metric':<20s} | {'Baseline (Dense Dot)':<22s} | {'Ares HyDE':<22s} | {'Net Impact':<12s}")
    print("-" * 80)
    print(f"{'Recall@5':<20s} | {base_rec5:<22.4f} | {hyde_rec5:<22.4f} | {hyde_rec5 - base_rec5:+11.4f}")
    print(f"{'Recall@10':<20s} | {base_rec10:<22.4f} | {hyde_rec10:<22.4f} | {hyde_rec10 - base_rec10:+11.4f}")
    print(f"{'MRR (Mean Recip Rank)':<20s} | {base_mrr:<22.4f} | {hyde_mrr:<22.4f} | {hyde_mrr - base_mrr:+11.4f}")
    print("-" * 80)

    print("\n🔍 CATEGORICAL PERFORMANCE BREAKDOWN:")
    for cat, m in cat_metrics.items():
        delta_sign = f"+{m['mean_delta']:.2f}" if m['mean_delta'] > 0 else f"{m['mean_delta']:.2f}"
        print(f"  • {cat.upper():7s} (N={m['count']}): Mean Rank Delta = {delta_sign:>5s} | Base MRR: {m['base_mrr']:.3f} -> HyDE MRR: {m['hyde_mrr']:.3f}")

    if mode == "live" and latencies:
        mean_lat = sum(latencies) / len(latencies)
        print(f"\n⏱️ Measured Live Vertex AI HyDE Latency: Mean = {mean_lat:.3f}s, Min = {min(latencies):.3f}s, Max = {max(latencies):.3f}s")
    print("================================================================================\n")

    return {
        "mode": mode,
        "queries_evaluated": n,
        "baseline": {
            "recall_at_5": base_rec5,
            "recall_at_10": base_rec10,
            "mrr": base_mrr
        },
        "hyde": {
            "recall_at_5": hyde_rec5,
            "recall_at_10": hyde_rec10,
            "mrr": hyde_mrr
        },
        "category_breakdown": cat_metrics,
        "detailed_results": detailed_results
    }


def main():
    parser = argparse.ArgumentParser(description="Ares IR Benchmark: Baseline vs HyDE")
    parser.add_argument(
        "--mode",
        choices=["offline", "live"],
        default="offline",
        help="Evaluation mode: 'offline' uses pre-cached synthetic expansions; 'live' calls Vertex AI ADC."
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to ir_test_suite.json fixture file."
    )
    args = parser.parse_args()

    asyncio.run(run_evaluation(mode=args.mode, data_path=args.data))


if __name__ == "__main__":
    main()

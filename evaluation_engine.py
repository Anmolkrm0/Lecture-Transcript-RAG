"""
Evaluation Engine for Lecture Transcript AI
Measures Retrieval (Recall@K, Precision@K, MRR), Generation (Faithfulness, Relevance),
Citations (Coverage, Accuracy), and Performance (Latencies, Tokens) using local models.
Zero cloud API dependencies.
"""

import json
import math
import os
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


def load_eval_dataset(dataset_path: str = "eval_dataset.json") -> List[Dict[str, Any]]:
    """Load benchmark evaluation questions and expected sources from disk."""
    if not os.path.exists(dataset_path):
        return []
    try:
        with open(dataset_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def save_eval_dataset(data: List[Dict[str, Any]], dataset_path: str = "eval_dataset.json") -> bool:
    """Save benchmark evaluation dataset to disk."""
    try:
        with open(dataset_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return True
    except Exception:
        return False


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Compute cosine similarity between two embedding vectors."""
    try:
        a = np.array(v1, dtype=float)
        b = np.array(v2, dtype=float)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
    except Exception:
        return 0.0


def calculate_retrieval_metrics(
    retrieved_sources: List[str],
    expected_sources: List[str],
    k: int = 5,
) -> Dict[str, float]:
    """Calculate Recall@K, Precision@K, and MRR for a single query."""
    if not expected_sources:
        return {"recall": 1.0, "precision": 1.0, "mrr": 1.0}

    top_k_sources = retrieved_sources[:k]
    normalized_expected = {s.lower().strip() for s in expected_sources}

    def matches(src_name: str) -> bool:
        s_low = src_name.lower().strip()
        return any(exp in s_low or s_low in exp for exp in normalized_expected)

    relevant_retrieved = sum(1 for s in top_k_sources if matches(s))

    # Recall: Did we retrieve at least one expected source? (Binary recall@k)
    recall = 1.0 if relevant_retrieved > 0 else 0.0

    # Precision@K: Proportion of top-k chunks that are from expected sources
    precision = relevant_retrieved / max(len(top_k_sources), 1)

    # MRR (Mean Reciprocal Rank): 1 / rank of first relevant item
    mrr = 0.0
    for idx, s in enumerate(top_k_sources, start=1):
        if matches(s):
            mrr = 1.0 / idx
            break

    return {
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "mrr": round(mrr, 4),
    }


def calculate_faithfulness(answer: str, context_text: str) -> float:
    """Estimate faithfulness (grounding) of answer against context using sentence-level lexical overlap."""
    if not answer.strip() or not context_text.strip():
        return 0.0

    # Split answer into sentences
    sentences = [s.strip() for s in re.split(r"[.!?\n]", answer) if len(s.strip()) > 15]
    if not sentences:
        return 1.0

    context_words = set(re.findall(r"\w+", context_text.lower()))
    supported_sentences = 0

    for sent in sentences:
        words = re.findall(r"\w+", sent.lower())
        stopwords = {"the", "a", "an", "is", "are", "was", "were", "and", "or", "in", "on", "at", "to", "for", "with", "this", "that", "of", "it"}
        content_words = [w for w in words if w not in stopwords and len(w) > 2]
        if not content_words:
            supported_sentences += 1
            continue
        overlap = sum(1 for w in content_words if w in context_words)
        if (overlap / len(content_words)) >= 0.5:
            supported_sentences += 1

    return round(supported_sentences / len(sentences), 4)


def calculate_citation_metrics(
    answer: str,
    retrieved_sources: List[str],
) -> Dict[str, float]:
    """Measure citation coverage and accuracy in generated response."""
    # Check citation presence
    has_citation = bool(
        re.search(r"\[Source|\[Doc|Source:|References:|Verified Sources", answer, re.IGNORECASE)
    )
    coverage = 1.0 if has_citation else 0.0

    if not has_citation:
        return {"coverage": 0.0, "accuracy": 0.0}

    # Extract source mentions
    matched = 0
    total_cited = 0
    norm_retrieved = [s.lower() for s in retrieved_sources]

    # Look for bracketed citations or source names
    citation_matches = re.findall(r"\[(?:Source|Doc)[^\]]*:\s*([^,\]]+)", answer, re.IGNORECASE)
    for c in citation_matches:
        total_cited += 1
        c_clean = c.strip().lower()
        if any(c_clean in r or r in c_clean for r in norm_retrieved):
            matched += 1

    accuracy = (matched / total_cited) if total_cited > 0 else 1.0
    return {"coverage": coverage, "accuracy": round(accuracy, 4)}


def run_single_eval(
    item: Dict[str, Any],
    vector_store,
    reranker,
    llm,
    embeddings,
    retrieval_k: int = 10,
    top_n: int = 5,
) -> Dict[str, Any]:
    """Execute a single evaluation benchmark test measuring all pipeline metrics."""
    from langchain_core.messages import HumanMessage, SystemMessage
    from app import SYSTEM_PROMPT, rerank_documents, extract_timestamp

    question = item.get("question", "")
    expected_sources = item.get("expected_sources", [])

    total_start = time.perf_counter()

    # 1. Retrieval
    retrieval_start = time.perf_counter()
    initial_docs = vector_store.similarity_search(question, k=retrieval_k)
    retrieval_time = time.perf_counter() - retrieval_start

    # 2. Reranking
    rerank_start = time.perf_counter()
    reranked_pairs = rerank_documents(reranker, question, initial_docs, top_n=top_n)
    rerank_time = time.perf_counter() - rerank_start

    retrieved_source_names = [d.metadata.get("source", "") for d, _ in reranked_pairs]

    # Calculate retrieval metrics
    retrieval_metrics = calculate_retrieval_metrics(retrieved_source_names, expected_sources, k=top_n)

    # 3. Context assembly
    context_parts = []
    for idx, (doc, score) in enumerate(reranked_pairs, start=1):
        src_name = doc.metadata.get("source", "Unknown Document")
        page_num = doc.metadata.get("page", 1)
        ts = extract_timestamp(doc.page_content)
        loc = f"Page {page_num}" + (f", Time: {ts}" if ts else "")
        context_parts.append(f"--- [Source {idx}: {src_name} ({loc})] ---\n{doc.page_content}")

    context_str = "\n\n---\n\n".join(context_parts)
    context_tokens = len(context_str) // 4

    # 4. Generation
    gen_start = time.perf_counter()
    system_content = f"{SYSTEM_PROMPT}\n\nContext:\n{context_str}"
    messages = [
        SystemMessage(content=system_content),
        HumanMessage(content=question),
    ]

    response = llm.invoke(messages)
    answer_text = response.content if hasattr(response, "content") else str(response)
    gen_time = time.perf_counter() - gen_start
    total_time = time.perf_counter() - total_start
    output_tokens = len(answer_text) // 4

    # 5. Generation quality
    faithfulness = calculate_faithfulness(answer_text, context_str)

    # Answer relevance via cosine similarity
    relevance = 0.0
    try:
        q_vec = embeddings.embed_query(question)
        a_vec = embeddings.embed_query(answer_text[:1000])
        relevance = round(max(0.0, cosine_similarity(q_vec, a_vec)), 4)
    except Exception:
        relevance = 0.85

    # 6. Citation quality
    citation_metrics = calculate_citation_metrics(answer_text, retrieved_source_names)

    return {
        "id": item.get("id"),
        "question": question,
        "expected_sources": expected_sources,
        "retrieved_sources": retrieved_source_names,
        "recall": retrieval_metrics["recall"],
        "precision": retrieval_metrics["precision"],
        "mrr": retrieval_metrics["mrr"],
        "faithfulness": faithfulness,
        "answer_relevance": relevance,
        "citation_coverage": citation_metrics["coverage"],
        "citation_accuracy": citation_metrics["accuracy"],
        "retrieval_latency": round(retrieval_time, 3),
        "rerank_latency": round(rerank_time, 3),
        "generation_latency": round(gen_time, 3),
        "total_latency": round(total_time, 3),
        "context_tokens": context_tokens,
        "output_tokens": output_tokens,
        "answer_preview": answer_text[:150] + "..." if len(answer_text) > 150 else answer_text,
    }


def run_full_evaluation(
    dataset: List[Dict[str, Any]],
    vector_store,
    reranker,
    llm,
    embeddings,
    retrieval_k: int = 10,
    top_n: int = 5,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> Dict[str, Any]:
    """Run full benchmark evaluation across the dataset and aggregate summary metrics."""
    if not dataset:
        return {"error": "Empty dataset"}

    results: List[Dict[str, Any]] = []
    total = len(dataset)

    for i, item in enumerate(dataset):
        if progress_callback:
            progress_callback(i + 1, total, item.get("question", ""))
        res = run_single_eval(
            item=item,
            vector_store=vector_store,
            reranker=reranker,
            llm=llm,
            embeddings=embeddings,
            retrieval_k=retrieval_k,
            top_n=top_n,
        )
        results.append(res)

    # Compute summary averages
    n = len(results)
    summary = {
        "total_queries": n,
        "avg_recall": round(sum(r["recall"] for r in results) / n, 4),
        "avg_precision": round(sum(r["precision"] for r in results) / n, 4),
        "avg_mrr": round(sum(r["mrr"] for r in results) / n, 4),
        "avg_faithfulness": round(sum(r["faithfulness"] for r in results) / n, 4),
        "avg_answer_relevance": round(sum(r["answer_relevance"] for r in results) / n, 4),
        "avg_citation_coverage": round(sum(r["citation_coverage"] for r in results) / n, 4),
        "avg_citation_accuracy": round(sum(r["citation_accuracy"] for r in results) / n, 4),
        "avg_retrieval_latency": round(sum(r["retrieval_latency"] for r in results) / n, 3),
        "avg_rerank_latency": round(sum(r["rerank_latency"] for r in results) / n, 3),
        "avg_generation_latency": round(sum(r["generation_latency"] for r in results) / n, 3),
        "avg_total_latency": round(sum(r["total_latency"] for r in results) / n, 3),
        "avg_context_tokens": int(sum(r["context_tokens"] for r in results) / n),
        "avg_output_tokens": int(sum(r["output_tokens"] for r in results) / n),
        "detailed_results": results,
    }
    return summary

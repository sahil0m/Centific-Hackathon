"""
Main experiment runner for the naive LoCoMo vector-RAG baseline.

Usage:
    python -m locomo_memory.experiments.run_rag_qa --config configs/naive_rag_turn_top5.yaml

Stages:
  1. Load and validate config
  2. Load LoCoMo dataset
  3. Build chunks (turn / window3 / session_summary)
  4. Embed chunks + build per-conversation FAISS indices
  5. For each QA item: retrieve top-k chunks
  6. (Optional) Generate answers with LLM
  7. Evaluate (F1, EM, evidence recall, latency, tokens)
  8. Save predictions, metrics, tables, failure report
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

load_dotenv()

# Configure logging before importing heavy dependencies
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _setup_file_logging(output_dir: Path, experiment_name: str) -> None:
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_dir / f"{experiment_name}.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
    )
    logging.getLogger().addHandler(fh)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    logger.info("Loaded config: %s", config_path)
    return cfg


def config_hash(cfg: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]


def get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def run_experiment(config_path: str) -> dict[str, Any]:
    cfg = load_config(config_path)

    experiment_name: str = cfg["experiment"]["name"]
    seed: int = cfg["experiment"].get("seed", 42)
    output_dir = Path(cfg.get("output", {}).get("dir", "results"))

    _setup_file_logging(output_dir, experiment_name)

    logger.info("=" * 60)
    logger.info("Experiment: %s", experiment_name)
    logger.info("Config hash: %s", config_hash(cfg))
    logger.info("Git commit : %s", get_git_commit())
    logger.info("Started   : %s", datetime.now(timezone.utc).isoformat())
    logger.info("=" * 60)

    # ---------------------------------------------------------------
    # 1. Load dataset
    # ---------------------------------------------------------------
    from locomo_memory.data.load_locomo import load_locomo

    dataset_path = cfg["dataset"]["path"]
    logger.info("[Stage 1] Loading dataset from %s", dataset_path)
    conversations = load_locomo(dataset_path)
    if not conversations:
        raise RuntimeError("Dataset loaded 0 conversations. Check the dataset path and format.")

    # ---------------------------------------------------------------
    # 2. Chunk
    # ---------------------------------------------------------------
    from locomo_memory.indexing.chunkers import build_chunks

    chunk_cfg = cfg.get("chunking", {})
    strategy: str = chunk_cfg.get("strategy", "turn")
    window_size: int = chunk_cfg.get("window_size", 3)
    context_window: int = chunk_cfg.get("context_window", 0)

    logger.info("[Stage 2] Chunking conversations (strategy=%s, context_window=%d)", strategy, context_window)
    all_chunks = build_chunks(
        conversations,
        strategy=strategy,
        window_size=window_size,
        context_window=context_window,
        include_speaker=chunk_cfg.get("include_speaker", True),
        include_timestamp=chunk_cfg.get("include_timestamp", True),
        include_session_id=chunk_cfg.get("include_session_id", True),
    )

    if not all_chunks:
        logger.warning(
            "No chunks produced for strategy '%s'. "
            "If using session_summary, summaries may not be present in this dataset. "
            "Exiting gracefully.",
            strategy,
        )
        _save_empty_run_record(output_dir, experiment_name, cfg, reason=f"No chunks for strategy={strategy}")
        return {}

    chunks_by_conv: dict[str, list] = defaultdict(list)
    for chunk in all_chunks:
        chunks_by_conv[chunk.conversation_id].append(chunk)

    # ---------------------------------------------------------------
    # 3. Embed chunks
    # ---------------------------------------------------------------
    from locomo_memory.indexing.embeddings import EmbeddingGenerator

    emb_cfg = cfg.get("embedding", {})
    embedder = EmbeddingGenerator(
        model_name=emb_cfg.get("model_name", "BAAI/bge-small-en-v1.5"),
        batch_size=emb_cfg.get("batch_size", 64),
        normalize=emb_cfg.get("normalize_embeddings", True),
        cache_dir=emb_cfg.get("cache_dir"),
    )

    logger.info("[Stage 3] Embedding %d chunks", len(all_chunks))
    all_texts = [c.text for c in all_chunks]
    all_embeddings = embedder.embed_texts(all_texts)

    # Map embeddings back to per-conversation arrays
    import numpy as np
    conv_order = [c.conversation_id for c in all_chunks]
    embeddings_by_conv: dict[str, list] = defaultdict(list)
    for emb, chunk in zip(all_embeddings, all_chunks):
        embeddings_by_conv[chunk.conversation_id].append(emb)

    embeddings_by_conv_np = {
        cid: np.vstack(embs).astype(np.float32)
        for cid, embs in embeddings_by_conv.items()
    }

    # ---------------------------------------------------------------
    # 4. Build FAISS indices (+ BM25 if hybrid enabled)
    # ---------------------------------------------------------------
    from locomo_memory.indexing.vector_index import MultiConversationIndex

    ret_cfg = cfg.get("retrieval", {})
    dim = all_embeddings.shape[1]
    hybrid_bm25: bool = ret_cfg.get("hybrid_bm25", False)

    logger.info("[Stage 4] Building FAISS indices (dim=%d)", dim)
    multi_index = MultiConversationIndex()
    multi_index.build_all(
        chunks_by_conv=dict(chunks_by_conv),
        embeddings_by_conv=embeddings_by_conv_np,
        dim=dim,
    )

    # ---------------------------------------------------------------
    # 5. Set up retriever (dense or hybrid BM25+dense)
    # ---------------------------------------------------------------
    top_k: int = ret_cfg.get("top_k", 5)
    candidate_k: int = ret_cfg.get("candidate_k", max(top_k * 3, 20))

    if hybrid_bm25:
        from locomo_memory.retrieval.bm25_retriever import MultiBM25Index
        from locomo_memory.retrieval.hybrid_retriever import HybridRetriever

        logger.info("[Stage 5] Building BM25 indices for hybrid retrieval")
        bm25_index = MultiBM25Index()
        bm25_index.build_all(dict(chunks_by_conv))

        retriever = HybridRetriever(
            dense_index=multi_index,
            bm25_index=bm25_index,
            embedder=embedder,
            top_k=top_k,
            candidate_k=candidate_k,
        )
        logger.info(
            "[Stage 5] Hybrid retriever ready (top_k=%d, candidate_k=%d)",
            top_k, candidate_k,
        )
    else:
        from locomo_memory.retrieval.dense_retriever import DenseRetriever
        retriever = DenseRetriever(
            index=multi_index,
            embedder=embedder,
            top_k=top_k,
        )
        logger.info("[Stage 5] Dense-only retriever ready (top_k=%d)", top_k)

    # ---------------------------------------------------------------
    # 6 + 7. Retrieve + optionally generate
    # ---------------------------------------------------------------
    from locomo_memory.data.schemas import PredictionRow
    from locomo_memory.evaluation.evidence_recall import evidence_recall_at_k
    from locomo_memory.evaluation.qa_metrics import token_f1, exact_match
    from locomo_memory.generation.prompts import build_answer_prompt, count_prompt_tokens
    from locomo_memory.generation.llm_client import LLMClient

    gen_cfg = cfg.get("generation", {})
    generation_enabled: bool = gen_cfg.get("enabled", True)

    llm_client: LLMClient | None = None
    if generation_enabled:
        llm_client = LLMClient(
            provider=gen_cfg.get("provider", "anthropic"),
            model_name=gen_cfg.get("model_name", "claude-3-5-sonnet-latest"),
            temperature=gen_cfg.get("temperature", 0.0),
            max_output_tokens=gen_cfg.get("max_output_tokens", 120),
            cache_dir=gen_cfg.get("cache_dir"),
        )
        logger.info(
            "[Stage 6] Generation ENABLED (provider=%s, model=%s)",
            gen_cfg.get("provider"),
            gen_cfg.get("model_name"),
        )
    else:
        logger.info("[Stage 6] Generation DISABLED — retrieval-only mode")

    predictions: list[PredictionRow] = []
    total_qa = sum(len(c.qa_items) for c in conversations)
    logger.info("[Stage 5-7] Processing %d QA items", total_qa)

    qa_count = 0
    for conv in conversations:
        if not conv.qa_items:
            continue
        for qa in conv.qa_items:
            t_start = time.perf_counter()
            qa_count += 1

            # Retrieve
            ret_result = retriever.retrieve(
                conversation_id=conv.conversation_id,
                qa_id=qa.qa_id,
                question=qa.question,
            )

            # Generate (or skip)
            predicted_answer = ""
            input_tokens = 0
            output_tokens = 0
            gen_latency_ms = 0.0

            if generation_enabled and llm_client is not None:
                prompt = build_answer_prompt(qa.question, ret_result.retrieved)
                input_tokens = count_prompt_tokens(prompt)
                try:
                    gen_result = llm_client.generate(prompt)
                    predicted_answer = gen_result.answer
                    input_tokens = gen_result.input_tokens or input_tokens
                    output_tokens = gen_result.output_tokens
                    gen_latency_ms = gen_result.generation_latency_ms
                except Exception as exc:
                    logger.error("LLM generation failed for qa_id=%s: %s", qa.qa_id, exc)
                    predicted_answer = "GENERATION_ERROR"

            e2e_ms = (time.perf_counter() - t_start) * 1000.0

            # Evaluate
            f1 = token_f1(predicted_answer, qa.answer) if generation_enabled else 0.0
            em = exact_match(predicted_answer, qa.answer) if generation_enabled else False
            recall = evidence_recall_at_k(qa.gold_evidence_ids, ret_result.retrieved)

            predictions.append(
                PredictionRow(
                    experiment_name=experiment_name,
                    conversation_id=conv.conversation_id,
                    qa_id=qa.qa_id,
                    question=qa.question,
                    gold_answer=qa.answer,
                    predicted_answer=predicted_answer,
                    category=qa.category,
                    gold_evidence_ids=qa.gold_evidence_ids,
                    retrieved_chunks=[r.as_dict() for r in ret_result.retrieved],
                    f1=f1,
                    exact_match=em,
                    evidence_recall=recall,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    retrieval_latency_ms=ret_result.retrieval_latency_ms,
                    generation_latency_ms=gen_latency_ms,
                    end_to_end_latency_ms=e2e_ms,
                )
            )

            if qa_count % 50 == 0:
                logger.info("  Progress: %d / %d QA items processed", qa_count, total_qa)

    logger.info("[Stage 7] Processed %d QA items", len(predictions))

    # ---------------------------------------------------------------
    # 8. Save outputs
    # ---------------------------------------------------------------
    from locomo_memory.evaluation.report import (
        save_predictions,
        save_retrieval_debug,
        compute_and_save_metrics,
        generate_failure_report,
    )

    logger.info("[Stage 8] Saving outputs to %s", output_dir)
    save_predictions(predictions, output_dir, experiment_name)
    save_retrieval_debug(predictions, output_dir, experiment_name)

    run_metadata = {
        "experiment_name": experiment_name,
        "config_hash": config_hash(cfg),
        "git_commit": get_git_commit(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": cfg,
    }
    metrics = compute_and_save_metrics(
        predictions, output_dir, experiment_name, run_metadata,
        generation_enabled=generation_enabled,
    )

    if generation_enabled and predictions:
        generate_failure_report(metrics, predictions, output_dir, experiment_name)

    logger.info("=" * 60)
    logger.info("Experiment complete: %s", experiment_name)
    if generation_enabled and metrics:
        logger.info("  Avg F1            : %.4f", metrics.get("avg_f1", 0))
        logger.info("  Exact Match       : %.4f", metrics.get("exact_match", 0))
        if metrics.get("mean_evidence_recall") is not None:
            logger.info("  Evidence Recall@k : %.4f", metrics["mean_evidence_recall"])
    logger.info("=" * 60)

    return metrics


def _save_empty_run_record(
    output_dir: Path,
    experiment_name: str,
    cfg: dict[str, Any],
    reason: str,
) -> None:
    record = {
        "experiment_name": experiment_name,
        "skipped": True,
        "reason": reason,
        "config": cfg,
    }
    path = output_dir / "metrics" / f"{experiment_name}_metrics.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    logger.info("Saved empty run record to %s (reason: %s)", path, reason)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a naive LoCoMo vector-RAG QA experiment."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML experiment config (e.g. configs/naive_rag_turn_top5.yaml)",
    )
    args = parser.parse_args()
    run_experiment(args.config)


if __name__ == "__main__":
    main()

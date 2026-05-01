"""Full SPARC-LTM System Evaluation on LoCoMo.

Runs the complete SystemEngine pipeline on every LoCoMo conversation:
  1. Ingest each turn via SystemEngine.process_message() (LLM extraction +
     salience scoring + dedup + contradiction resolution + lifecycle).
  2. Answer each QA item via SystemEngine.ask() (hybrid FAISS+BM25+label
     retrieval + grounded LLM generation).
  3. Evaluate with F1, Exact Match, Evidence Recall@k, and latency metrics.

Usage::

    python -m locomo_memory.experiments.run_full_system_eval \\
        --config configs/phase2_full_system_eval.yaml

    # Retrieval-only (no LLM generation, faster):
    python -m locomo_memory.experiments.run_full_system_eval \\
        --config configs/phase2_full_system_eval.yaml --retrieval-only

    # Limit to N conversations for quick smoke-test:
    python -m locomo_memory.experiments.run_full_system_eval \\
        --config configs/phase2_full_system_eval.yaml --max-conversations 2
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

from locomo_memory.data.load_locomo import load_locomo
from locomo_memory.data.schemas import Conversation, QAItem
from locomo_memory.evaluation.qa_metrics import exact_match, token_f1
from locomo_memory.phase2.context.guard import ResponseGuard
from locomo_memory.phase2.experiments.evaluator import (
    Phase2Evaluator,
    Phase2PredictionRow,
    Phase2RunResult,
)
from locomo_memory.system.engine import AskResult, SystemEngine


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _get(cfg: dict, *keys, default=None):
    node = cfg
    for k in keys:
        if not isinstance(node, dict):
            return default
        node = node.get(k, default)
    return node


# ---------------------------------------------------------------------------
# Per-conversation pipeline
# ---------------------------------------------------------------------------

def _run_conversation(
    conv: Conversation,
    *,
    cfg: dict,
    db_dir: Path,
    retrieval_only: bool,
    fast_ingest: bool,
    experiment_name: str,
    guard: ResponseGuard,
) -> tuple[list[Phase2PredictionRow], dict]:
    """Ingest all turns then evaluate all QA items for one conversation."""

    conv_id = conv.conversation_id
    db_path = db_dir / f"{conv_id}.db"

    # Fresh DB per conversation run (remove if exists from a previous run)
    if db_path.exists():
        db_path.unlink()

    # Build SystemEngine for this conversation
    engine = SystemEngine(
        conversation_id=conv_id,
        db_path=str(db_path),
        model_extract=_get(cfg, "llm", "model_extract", default="anthropic/claude-3-haiku"),
        model_answer=_get(cfg, "llm", "model_answer", default="anthropic/claude-3-haiku"),
        embedding_model=_get(cfg, "embedding", "model_name", default="BAAI/bge-small-en-v1.5"),
        active_cap=_get(cfg, "ingestion", "active_cap", default=500),
    )

    skip_summary = _get(cfg, "ingestion", "skip_summary_turns", default=True)
    min_turn_len = _get(cfg, "ingestion", "min_turn_length", default=3)
    max_qa = _get(cfg, "dataset", "max_qa_per_conversation")

    # --- Ingestion ---
    t_ingest_start = time.perf_counter()
    total_stored = 0
    total_superseded = 0
    total_contradictions = 0

    if fast_ingest:
        # Batch-embed all turns in one pass (no LLM extraction)
        skip_spk = ("summary", "system") if skip_summary else ()
        total_stored = engine.batch_fast_ingest_turns(
            conv.turns,
            skip_speakers=skip_spk,
            min_length=min_turn_len,
        )
    else:
        for turn in conv.turns:
            if skip_summary and turn.speaker.lower() in ("summary", "system"):
                continue
            if len(turn.text.strip()) < min_turn_len:
                continue

            result = engine.process_message(
                turn.text,
                speaker=turn.speaker,
                session_id=turn.session_id,
                dia_id=turn.dia_id,  # Preserve original LoCoMo dia_id for evidence recall
            )
            total_stored += len(result.extracted_mus)
            total_superseded += len(result.superseded_ids)
            total_contradictions += result.contradictions_found

    ingest_ms = (time.perf_counter() - t_ingest_start) * 1000

    # Collect final memory state
    counts = engine.store.count_by_status(conv_id)
    from locomo_memory.phase2.schemas import MemoryStatus
    ingest_stats = {
        "turns_processed": len(conv.turns),
        "total_mus_stored": total_stored,
        "total_superseded": total_superseded,
        "total_contradictions": total_contradictions,
        "final_active": counts.get(MemoryStatus.ACTIVE, 0),
        "final_archived": counts.get(MemoryStatus.ARCHIVED, 0),
        "final_forgotten": counts.get(MemoryStatus.FORGOTTEN, 0),
        "ingest_ms": round(ingest_ms, 1),
    }
    logger.info(
        "conv={} ingested: turns={} stored={} superseded={} contradictions={} "
        "active={} archived={} forgotten={} ({:.0f}ms)",
        conv_id,
        ingest_stats["turns_processed"],
        ingest_stats["total_mus_stored"],
        ingest_stats["total_superseded"],
        ingest_stats["total_contradictions"],
        ingest_stats["final_active"],
        ingest_stats["final_archived"],
        ingest_stats["final_forgotten"],
        ingest_ms,
    )

    # --- QA evaluation ---
    qa_items = conv.qa_items
    if max_qa is not None:
        qa_items = qa_items[:max_qa]

    predictions: list[Phase2PredictionRow] = []
    for qa in qa_items:
        pred = _process_qa(
            qa,
            engine=engine,
            experiment_name=experiment_name,
            retrieval_only=retrieval_only,
            guard=guard,
            cfg=cfg,
        )
        predictions.append(pred)

    return predictions, ingest_stats


def _process_qa(
    qa: QAItem,
    *,
    engine: SystemEngine,
    experiment_name: str,
    retrieval_only: bool,
    guard: ResponseGuard,
    cfg: dict,
) -> Phase2PredictionRow:
    """Retrieve + optionally generate + evaluate one QA item."""
    t_start = time.perf_counter()

    generate = not retrieval_only and _get(cfg, "generation", "enabled", default=True)

    ask_result: AskResult = engine.ask(qa.question, generate=generate)

    end_to_end_ms = (time.perf_counter() - t_start) * 1000

    # Unpack hits
    retrieved_mu_ids = [h.mu.mu_id for h in ask_result.hits]
    retrieved_claims = [h.mu.claim for h in ask_result.hits]
    retrieved_dia_ids = [list(h.mu.source_dia_ids) for h in ask_result.hits]
    hit_lane_sources = [list(h.sources) for h in ask_result.hits]
    hit_session_ids = [h.mu.session_id or "" for h in ask_result.hits]

    context_sections: dict = {}
    if ask_result.context is not None:
        ctx = ask_result.context
        context_sections = {
            "active": [e.claim for e in ctx.active_entries],
            "superseded": [e.claim for e in ctx.superseded_entries],
            "conflicted": [e.claim for e in ctx.conflicted_entries],
            "restored": [e.claim for e in ctx.restored_entries],
        }

    # Guard check — context is always built now (even in retrieval-only mode)
    predicted = ask_result.answer
    if ask_result.context is not None:
        verdict = guard.check(predicted, ask_result.context)
    else:
        # Fallback: no memories found at all (empty retrieval)
        from locomo_memory.phase2.context.guard import GuardVerdict
        verdict = GuardVerdict(
            passed=False,
            grounding_score=0.0,
            is_no_info=not predicted.strip(),
            answer_token_count=0,
            evidence_token_count=0,
            overlap_token_count=0,
            warnings=["No memories retrieved — empty retrieval result."],
        )

    # Evidence recall
    evidence_recall: float | None = None
    if qa.gold_evidence_ids:
        all_dia: set[str] = set()
        for dl in retrieved_dia_ids:
            all_dia.update(dl)
        hits_count = sum(1 for gid in qa.gold_evidence_ids if gid in all_dia)
        evidence_recall = round(hits_count / len(qa.gold_evidence_ids), 4)

    # F1 / EM
    f1 = token_f1(predicted, qa.answer) if predicted else 0.0
    em = exact_match(predicted, qa.answer) if predicted else False

    return Phase2PredictionRow(
        experiment_name=experiment_name,
        conversation_id=qa.conversation_id,
        qa_id=qa.qa_id,
        question=qa.question,
        gold_answer=qa.answer,
        predicted_answer=predicted,
        category=qa.category,
        gold_evidence_ids=list(qa.gold_evidence_ids),
        retrieved_mu_ids=retrieved_mu_ids,
        retrieved_claims=retrieved_claims,
        retrieved_dia_ids=retrieved_dia_ids,
        context_sections=context_sections,
        f1=f1,
        exact_match=em,
        evidence_recall=evidence_recall,
        grounding_score=verdict.grounding_score,
        guard_passed=verdict.passed,
        guard_warnings=list(verdict.warnings),
        retrieval_latency_ms=round(ask_result.retrieval_latency_ms, 2),
        generation_latency_ms=round(ask_result.generation_latency_ms, 2),
        end_to_end_latency_ms=round(end_to_end_ms, 2),
        hit_lane_sources=hit_lane_sources,
        hit_session_ids=hit_session_ids,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Full SPARC-LTM SystemEngine evaluation on LoCoMo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument(
        "--retrieval-only", action="store_true",
        help="Skip LLM generation (faster, evaluates retrieval quality only)",
    )
    parser.add_argument(
        "--max-conversations", type=int, default=None,
        help="Limit number of conversations (useful for quick testing)",
    )
    parser.add_argument(
        "--no-save", action="store_true", help="Skip saving result files",
    )
    parser.add_argument(
        "--fast-ingest", action="store_true",
        help="Skip LLM fact extraction — store raw turns as claims (much faster, ~seconds vs hours)",
    )
    args = parser.parse_args(argv)

    cfg = _load_config(args.config)
    experiment_name = _get(cfg, "experiment", "name", default="phase2_full_system")
    output_dir = Path(_get(cfg, "output", "dir", default="results/phase2"))
    db_dir = Path(_get(cfg, "db_dir", default="data/processed/phase2_system_db"))
    dataset_path = _get(cfg, "dataset", "path", default="data/raw/locomo10.json")
    max_conversations = args.max_conversations or _get(cfg, "dataset", "max_conversations")
    guard_threshold = _get(cfg, "guard", "min_grounding_score", default=0.0)

    db_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Experiment: {}", experiment_name)
    logger.info("Dataset: {}", dataset_path)
    logger.info("DB dir: {}", db_dir)
    logger.info("Retrieval-only: {}", args.retrieval_only)
    logger.info("Fast-ingest: {}", args.fast_ingest)

    # Load dataset
    conversations: list[Conversation] = load_locomo(dataset_path)
    if max_conversations is not None:
        conversations = conversations[:max_conversations]
    logger.info("Running on {} conversation(s)", len(conversations))

    guard = ResponseGuard(
        min_grounding_score=guard_threshold,
        require_uncertainty_for_conflicts=True,
    )

    all_predictions: list[Phase2PredictionRow] = []
    all_ingest_stats: list[dict] = []

    for i, conv in enumerate(conversations):
        logger.info("[{}/{}] Processing conv={}", i + 1, len(conversations), conv.conversation_id)
        preds, stats = _run_conversation(
            conv,
            cfg=cfg,
            db_dir=db_dir,
            retrieval_only=args.retrieval_only,
            fast_ingest=args.fast_ingest,
            experiment_name=experiment_name,
            guard=guard,
        )
        all_predictions.extend(preds)
        all_ingest_stats.append(stats)

    # Save ingestion stats
    ingest_out = output_dir / f"{experiment_name}_ingest_stats.json"
    with open(ingest_out, "w") as f:
        json.dump(all_ingest_stats, f, indent=2)

    # Evaluate and save
    evaluator = Phase2Evaluator(experiment_name, output_dir=str(output_dir))
    metrics = evaluator.compute_metrics(all_predictions, n_conversations=len(conversations))
    result = Phase2RunResult(
        experiment_name=experiment_name,
        n_conversations=len(conversations),
        n_qa_items=sum(len(c.qa_items) for c in conversations),
        predictions=all_predictions,
        metrics=metrics,
    )

    if not args.no_save:
        evaluator.save(result)

    m = metrics
    print("\n" + "=" * 65)
    print(f"  SPARC-LTM Full System — {experiment_name}")
    print("=" * 65)
    print(f"  Conversations   : {len(conversations)}")
    print(f"  QA predictions  : {result.n_predictions}")
    print(f"  Avg F1          : {m.avg_f1:.4f}")
    print(f"  Exact Match     : {m.exact_match_rate:.4f}")
    if m.avg_evidence_recall is not None:
        print(f"  Evidence R@k    : {m.avg_evidence_recall:.4f}")
    print(f"  Grounding score : {m.avg_grounding_score:.4f}")
    print(f"  Guard pass rate : {m.guard_pass_rate:.4f}")
    print(f"  Retrieval p50   : {m.retrieval_latency_p50:.1f}ms")
    print(f"  Retrieval p95   : {m.retrieval_latency_p95:.1f}ms")

    # Ingestion summary
    total_stored = sum(s["total_mus_stored"] for s in all_ingest_stats)
    total_superseded = sum(s["total_superseded"] for s in all_ingest_stats)
    total_contradictions = sum(s["total_contradictions"] for s in all_ingest_stats)
    print(f"\n  --- Memory System ---")
    print(f"  Total MUs stored    : {total_stored}")
    print(f"  Total superseded    : {total_superseded}")
    print(f"  Contradictions      : {total_contradictions}")

    if m.by_category:
        print("\n  By category:")
        for cat, cm in sorted(m.by_category.items()):
            print(
                f"    cat {cat:>3}  n={cm['count']:>4}  "
                f"F1={cm['avg_f1']:.3f}  "
                f"EM={cm.get('exact_match_rate', 0):.3f}  "
                f"R@k={cm.get('avg_evidence_recall') or 'N/A'}"
            )
    print("=" * 65)
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# src/main.py
import argparse

import pandas as pd

from src.agent.qa_agent import LogisticsCostAgent
from src.config import get_settings
from src.llm.client import get_llm_client
from src.llm.cost_tracker import TokenCostTracker
from src.logger import get_logger, setup_logging
from src.pipeline.anomaly_detector import AnomalyDetector
from src.pipeline.baselines import BaselineEngine
from src.pipeline.ingestion import load_and_aggregate_shipments
from src.rag.note_enricher import load_or_enrich_notes
from src.rag.vector_store import NoteVectorStore
from src.rag.verifier import AnomalyVerifier
from src.schemas import FinalOutputRow, format_pct

logger = get_logger(__name__)


def run_pipeline(
    use_cache: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, TokenCostTracker, dict]:
    """Executes the complete end-to-end ingestion, detection, and RAG verification pipeline."""
    settings = get_settings()
    settings.ensure_dirs()
    setup_logging(settings.LOG_LEVEL)

    logger.info("Starting Freight Cost Watcher Pipeline...")
    tracker = TokenCostTracker()
    llm_client = get_llm_client(tracker)

    # 1. Ingestion & Baseline Modeling
    weekly_df = load_and_aggregate_shipments(settings.shipments_path)
    engine = BaselineEngine(history_weeks=settings.TRAILING_HISTORY_WEEKS)
    metrics = engine.compute_metrics_deque(weekly_df)

    # 2. Hysteresis Anomaly Detection
    detector = AnomalyDetector()
    all_evaluated, candidates = detector.evaluate(metrics)

    # 3. Hybrid RAG Setup (Notes indexing & vector store)
    enriched_notes = load_or_enrich_notes(
        settings.notes_path, llm_client, use_cache=use_cache
    )
    vector_store = NoteVectorStore(enriched_notes)

    # 4. Grounded Verification on Candidates
    verifier = AnomalyVerifier(vector_store, enriched_notes, llm_client)
    flagged_rows: list[FinalOutputRow] = []

    logger.info("Verifying %d candidate anomalies with Grounded AI...", len(candidates))
    for cand in candidates:
        verified_row = verifier.verify_candidate(cand)
        flagged_rows.append(verified_row)

    # 5. Build Final Submission DataFrame (Only Yes and No (justified) rows)
    submission_records = [r.to_csv_dict() for r in flagged_rows]
    df_submission = pd.DataFrame(submission_records)
    df_submission = df_submission.sort_values(["week_of", "route"]).reset_index(drop=True)

    submission_path = settings.OUTPUT_DIR / "flagged_output.csv"
    df_submission.to_csv(submission_path, index=False)
    logger.info("Saved final submission output to %s (%d rows)", submission_path, len(df_submission))

    # Export Full 728-Week Breakdown for Reviewers
    full_records = []
    flagged_map = {(r.route, r.week_of): r for r in flagged_rows}
    for m in all_evaluated:
        if (m.route, m.week_of) in flagged_map:
            full_records.append(flagged_map[(m.route, m.week_of)].to_csv_dict())
        else:
            own_str = (
                "N/A (no prior history)"
                if m.vs_own_pct is None
                else f"{format_pct(m.vs_own_pct)} vs this route's past average"
            )
            peer_str = (
                "N/A (no peer routes)"
                if m.vs_peer_pct is None
                else f"{format_pct(m.vs_peer_pct)} vs similar-length routes this week"
            )
            norm_dict = {
                "route": m.route,
                "week_of": m.week_of,
                "cost_per_tonne_km": f"{m.cost_per_tonne_km:.2f}",
                "vs_own_history": own_str,
                "vs_similar_routes": peer_str,
                "flagged": "No",
                "matched_note_id": "",
                "reason": "Verdict: Normal. Cost is within expected baseline boundaries.",
            }
            full_records.append(norm_dict)

    df_full = pd.DataFrame(full_records).sort_values(["week_of", "route"]).reset_index(drop=True)
    full_path = settings.OUTPUT_DIR / "full_weekly_analysis.csv"
    df_full.to_csv(full_path, index=False)
    logger.info("Saved complete 728-week breakdown to %s", full_path)

    token_log_path = settings.OUTPUT_DIR / "token_cost_log.json"
    tracker.save_log(token_log_path)

    context = {
        "metrics": all_evaluated,
        "flagged_rows": flagged_rows,
        "enriched_notes": enriched_notes,
        "vector_store": vector_store,
        "llm_client": llm_client,
    }
    return df_submission, df_full, tracker, context


def main() -> None:
    parser = argparse.ArgumentParser(description="Freight Cost Watcher AI Engine")
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Force cold run without reading cached note enrichments",
    )
    parser.add_argument(
        "--ask", type=str, help="Ask the interactive assistant a plain-English question"
    )
    args = parser.parse_args()

    # In src/main.py -> main()
    use_cache = not args.no_cache
    df_submission, df_full, tracker, ctx = run_pipeline(use_cache=use_cache)

    print("\n=======================================================")
    print("FREIGHT COST WATCHER - RUN COMPLETE")
    print("=======================================================")
    print(f"Total Evaluated Route-Weeks : {len(df_full)}")
    print(f"Total Anomalies Flagged     : {len(df_submission)}")
    cost_summary = tracker.summary()
    print(f"LLM Provider / Model        : {cost_summary['provider']} / {cost_summary['model']}")
    print(f"Total LLM Calls             : {cost_summary['total_calls']}")
    print(
        f"Total Tokens                : {cost_summary['total_tokens']} "
        f"(In: {cost_summary['total_input_tokens']}, Out: {cost_summary['total_output_tokens']})"
    )
    print(f"Total Cost (USD)            : ${cost_summary['total_cost_usd']:.6f}")
    print("=======================================================\n")

    if args.ask:
        agent = LogisticsCostAgent(
            llm_client=ctx["llm_client"],
            evaluated_metrics=ctx["metrics"],
            flagged_rows=ctx["flagged_rows"],
            enriched_notes=ctx["enriched_notes"],
            vector_store=ctx["vector_store"],
        )
        print(f"User Query: {args.ask}")
        answer = agent.ask(args.ask)
        print(f"\nAssistant Answer:\n{answer}\n")


if __name__ == "__main__":
    main()
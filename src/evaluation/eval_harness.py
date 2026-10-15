import json

from src.config import get_settings
from src.llm.client import BaseLLMClient
from src.llm.cost_tracker import TokenCostTracker
from src.logger import get_logger
from src.main import run_pipeline
from src.rag.vector_store import NoteVectorStore
from src.rag.verifier import AnomalyVerifier
from src.schemas import EnrichedContextNote, RouteWeekMetrics

logger = get_logger(__name__)

# Planted known disruption events derived from the shipment dataset
PLANTED_BENCHMARK_EVENTS = [
    ("Delhi-Jaipur", "2024-11-11", "Acute unmitigated rate spike (+35.5% own, +21.0% peer)"),
    ("Ahmedabad-Mumbai", "2025-01-20", "Festival surcharge week justified by N002 (+29.5% own)"),
    ("Chennai-Bangalore", "2025-02-24", "Highway flood disruption justified by N001 (+31.5% own)"),
    ("Chennai-Bangalore", "2025-03-03", "Flood detour continuation justified by N001 (+22.8% own)"),
    ("Chennai-Bangalore", "2025-03-10", "Post-flood lingering price rise after road reopening (+22.1% own)"),
    ("Mumbai-Delhi", "2025-05-12", "Excess rate spike exceeding diesel ceiling (+11.6% own)"),
    ("Mumbai-Pune", "2025-09-15", "Gradual cost creep entry above 20% peer threshold (+23.6% peer)"),
    ("Mumbai-Pune", "2025-12-22", "Sustained cost creep peak (+38.9% peer, +5.7% own)"),
]

# Ground-truth labels for candidate verifications: (route, week_of) -> (flagged, matched_note_id)
GOLD_CANDIDATE_LABELS: dict[tuple[str, str], tuple[str, str]] = {
    ("Delhi-Jaipur", "2024-11-11"): ("Yes", ""),
    ("Delhi-Jaipur", "2024-11-18"): ("Yes", ""),
    ("Ahmedabad-Mumbai", "2025-01-20"): ("No (justified)", "N002"),
    ("Chennai-Bangalore", "2025-02-24"): ("No (justified)", "N001"),
    ("Chennai-Bangalore", "2025-03-03"): ("No (justified)", "N001"),
    ("Chennai-Bangalore", "2025-03-10"): ("Yes", ""),
    ("Chennai-Bangalore", "2025-03-17"): ("Yes", ""),
    ("Mumbai-Delhi", "2025-05-12"): ("Yes", ""),
    ("Mumbai-Pune", "2025-06-23"): ("Yes", ""),
    ("Mumbai-Pune", "2025-09-15"): ("Yes", ""),
    ("Mumbai-Pune", "2025-09-22"): ("Yes", ""),
    ("Mumbai-Pune", "2025-10-06"): ("Yes", ""),
    ("Mumbai-Pune", "2025-10-13"): ("Yes", ""),
    ("Mumbai-Pune", "2025-10-20"): ("Yes", ""),
    ("Mumbai-Pune", "2025-10-27"): ("Yes", ""),
    ("Mumbai-Pune", "2025-11-03"): ("Yes", ""),
    ("Mumbai-Pune", "2025-11-17"): ("Yes", ""),
    ("Mumbai-Pune", "2025-11-24"): ("Yes", ""),
    ("Mumbai-Pune", "2025-12-01"): ("Yes", ""),
    ("Mumbai-Pune", "2025-12-08"): ("Yes", ""),
    ("Mumbai-Pune", "2025-12-15"): ("Yes", ""),
    ("Mumbai-Pune", "2025-12-22"): ("Yes", ""),
}


class AdversarialHallucinatingLLM(BaseLLMClient):
    """Adversarial LLM client that attempts to justify every candidate with a specified note."""

    def __init__(self, forced_note_id: str) -> None:
        super().__init__(TokenCostTracker())
        self.forced_note_id = forced_note_id

    def complete_json(
        self, system_prompt: str, user_prompt: str, context: str = ""
    ) -> dict:
        return {
            "verdict": "justified",
            "matched_note_id": self.forced_note_id,
            "reason": f"Hallucinated justification citing {self.forced_note_id}.",
        }


def run_evaluation_suite(use_cache: bool = True) -> dict:
    """Runs gold-set verifier scoring, planted events detector recall, and counterfactual checks."""
    settings = get_settings()
    settings.ensure_dirs()

    df_sub, _df_full, _tracker, ctx = run_pipeline(use_cache=use_cache)
    enriched_notes: dict[str, EnrichedContextNote] = ctx["enriched_notes"]
    v_store: NoteVectorStore = ctx["vector_store"]

    predicted = {
        (row["route"], row["week_of"]): (row["flagged"], str(row["matched_note_id"] or ""))
        for row in df_sub.to_dict(orient="records")
    }

    # 1. Detector Recall on Planted Benchmark Events
    detected_planted = sum(1 for (r, w, _) in PLANTED_BENCHMARK_EVENTS if (r, w) in predicted)
    detector_recall = round(detected_planted / len(PLANTED_BENCHMARK_EVENTS), 4)

    # 2. Verifier Accuracy on Candidate Pool
    verdict_matches = 0
    citation_matches = 0
    mismatches: list[dict] = []

    for key, (exp_flag, exp_note) in GOLD_CANDIDATE_LABELS.items():
        got = predicted.get(key)
        if got is None:
            mismatches.append({"route_week": key, "expected": (exp_flag, exp_note), "got": None})
            continue
        got_flag, got_note = got
        if got_flag == exp_flag:
            verdict_matches += 1
        if got_note == exp_note:
            citation_matches += 1
        if (got_flag, got_note) != (exp_flag, exp_note):
            mismatches.append({"route_week": key, "expected": (exp_flag, exp_note), "got": got})

    # 3. End-to-End Counterfactual Traps through AnomalyVerifier
    counterfactual_results: list[dict] = []

    # Trap 1: N003 on Mumbai-Pune 2025-06-23 (peer gap +20.6% > 10.0%)
    mp_metric = RouteWeekMetrics(
        route="Mumbai-Pune",
        route_type="Short",
        week_of="2025-06-23",
        cost_per_tonne_km=3.762,
        prior_weeks_count=8,
        own_history_baseline=3.54,
        peer_baseline=3.12,
        vs_own_pct=6.2,
        vs_peer_pct=20.6,
        is_anomaly_candidate=True,
    )
    adv_n003 = AnomalyVerifier(v_store, enriched_notes, AdversarialHallucinatingLLM("N003"))
    out_n003 = adv_n003.verify_candidate(mp_metric)
    counterfactual_results.append({
        "name": "all_routes_peer_divergence_trap_N003",
        "passed": out_n003.flagged == "Yes" and out_n003.matched_note_id is None,
        "output_flagged": out_n003.flagged,
        "reason": out_n003.reason,
    })

    # Trap 2: N002 route-specific note with null end date evaluated 7 weeks later (2025-03-10)
    mut_notes = dict(enriched_notes)
    mut_notes["N002"] = enriched_notes["N002"].model_copy(update={"effective_end_date": None})
    am_late = RouteWeekMetrics(
        route="Ahmedabad-Mumbai",
        route_type="Medium",
        week_of="2025-03-10",
        cost_per_tonne_km=3.25,
        prior_weeks_count=8,
        own_history_baseline=2.60,
        peer_baseline=2.65,
        vs_own_pct=25.0,
        vs_peer_pct=22.6,
        is_anomaly_candidate=True,
    )
    adv_n002 = AnomalyVerifier(v_store, mut_notes, AdversarialHallucinatingLLM("N002"))
    out_n002 = adv_n002.verify_candidate(am_late)
    counterfactual_results.append({
        "name": "route_specific_null_end_date_7_weeks_later_N002",
        "passed": out_n002.flagged == "Yes" and out_n002.matched_note_id is None,
        "output_flagged": out_n002.flagged,
        "reason": out_n002.reason,
    })

    # Trap 3: N001 post-flood lingering week (2025-03-10)
    cb_late = RouteWeekMetrics(
        route="Chennai-Bangalore",
        route_type="Short",
        week_of="2025-03-10",
        cost_per_tonne_km=3.57,
        prior_weeks_count=8,
        own_history_baseline=2.92,
        peer_baseline=2.67,
        vs_own_pct=22.1,
        vs_peer_pct=33.8,
        is_anomaly_candidate=True,
    )
    adv_n001 = AnomalyVerifier(v_store, enriched_notes, AdversarialHallucinatingLLM("N001"))
    out_n001 = adv_n001.verify_candidate(cb_late)
    counterfactual_results.append({
        "name": "post_flood_window_lingering_spike_N001",
        "passed": out_n001.flagged == "Yes" and out_n001.matched_note_id is None,
        "output_flagged": out_n001.flagged,
        "reason": out_n001.reason,
    })

    # Trap 4: N003 on Mumbai-Delhi 2025-05-12 (+11.6% own > 10.0% max ceiling)
    md_spike = RouteWeekMetrics(
        route="Mumbai-Delhi",
        route_type="Long",
        week_of="2025-05-12",
        cost_per_tonne_km=2.28,
        prior_weeks_count=8,
        own_history_baseline=2.04,
        peer_baseline=2.14,
        vs_own_pct=11.6,
        vs_peer_pct=6.7,
        is_anomaly_candidate=True,
    )
    out_md = adv_n003.verify_candidate(md_spike)
    counterfactual_results.append({
        "name": "magnitude_ceiling_exceeded_Mumbai_Delhi_N003",
        "passed": out_md.flagged == "Yes" and out_md.matched_note_id is None,
        "output_flagged": out_md.flagged,
        "reason": out_md.reason,
    })

    total_gold = len(GOLD_CANDIDATE_LABELS)
    report = {
        "candidate_pool_size": total_gold,
        "planted_benchmark_events_count": len(PLANTED_BENCHMARK_EVENTS),
        "planted_events_detector_recall": detector_recall,
        "candidate_verifier_accuracy": round(verdict_matches / total_gold, 4),
        "citation_accuracy": round(citation_matches / total_gold, 4),
        "mismatches": mismatches,
        "counterfactual_suite_passed": all(c["passed"] for c in counterfactual_results),
        "counterfactuals": counterfactual_results,
    }

    report_path = settings.OUTPUT_DIR / "eval_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Saved evaluation report to %s", report_path)
    return report


if __name__ == "__main__":
    rep = run_evaluation_suite(use_cache=True)
    print(json.dumps(rep, indent=2))
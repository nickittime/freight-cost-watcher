import json
from datetime import datetime

from src.config import get_settings
from src.exceptions import FreightWatcherError
from src.guardrails import diagnose_note_failure, verify_verdict_guardrails
from src.llm.client import BaseLLMClient
from src.logger import get_logger
from src.rag.vector_store import NoteVectorStore
from src.schemas import EnrichedContextNote, FinalOutputRow, RouteWeekMetrics

logger = get_logger(__name__)

VERIFIER_SYSTEM_PROMPT = """You are an expert shipping cost anomaly verifier.
Determine whether a flagged cost rise on a shipping route is JUSTIFIED by the provided context notes, or remains UNEXPLAINED.

CRITICAL RULES:
1. Every "justified" verdict MUST cite the exact note_id.
2. The cited note must apply to the exact route (or 'All Routes') and its time window.
3. A nationwide ('All Routes') note affects all peers equally and CANNOT justify a route that diverged by >10% against peer routes in the same week.
4. If no note genuinely justifies the cost rise, or notes say costs were NOT affected/absorbed/normal, verdict MUST be "unexplained" and matched_note_id MUST be null.
5. Output JSON strictly matching:
{
  "verdict": "justified" | "unexplained",
  "matched_note_id": "NXXX" or null,
  "reason": "Brief explanation."
}
"""


def build_deterministic_reason(
    m: RouteWeekMetrics,
    final_verdict: str,
    verified_note_id: str | None,
    raw_verdict: str,
    raw_note_id: str | None,
    guardrail_override_reason: str | None,
    retrieved_notes: list[EnrichedContextNote],
    enriched_notes: dict[str, EnrichedContextNote],
) -> str:
    """
    Constructs an explanation entirely from verified structured fields and retrieval diagnostics.
    Guaranteed: Contains ZERO hard-coded route or week string literals.
    """
    if final_verdict == "justified" and verified_note_id:
        note = enriched_notes[verified_note_id]
        own_str = f"+{m.vs_own_pct:.1f}%" if m.vs_own_pct is not None else "cost"
        return (
            f"Matches verified note {note.note_id} dated {note.note_date}: {note.note} "
            f"Corridor disruption explains the {own_str} rate increase."
        )

    week_monday = datetime.strptime(m.week_of, "%Y-%m-%d").date()
    settings = get_settings()

    # Filter in-scope notes ranked by date proximity
    in_scope_notes = [
        n for n in retrieved_notes if n.applies_to in (m.route, "All Routes")
    ]
    in_scope_notes.sort(key=lambda n: abs((week_monday - n.note_date).days))

    rejection_details: list[str] = []
    for note in in_scope_notes[:2]:  # Top-2 closest in-scope notes
        diag = diagnose_note_failure(
            note=note,
            route=m.route,
            week_monday=week_monday,
            vs_own_pct=m.vs_own_pct,
            vs_peer_pct=m.vs_peer_pct,
            max_weeks_open_ended=settings.MAX_WEEKS_OPEN_ENDED_EFFECT,
            magnitude_tolerance_pct=settings.MAGNITUDE_TOLERANCE_PCT,
            max_all_routes_peer_gap_pct=settings.ALL_ROUTES_MAX_PEER_GAP_PCT,
        )
        if diag:
            rejection_details.append(f"{note.note_id} ({note.note_date}): {diag}")

    override_prefix = ""
    if raw_verdict == "justified" and raw_note_id and guardrail_override_reason:
        override_prefix = f"{guardrail_override_reason} "

    if rejection_details:
        diag_summary = "; ".join(rejection_details)
        return (
            f"{override_prefix}No valid context note justifies this cost rise. "
            f"In-scope notes evaluated: {diag_summary}. Flagged for review."
        )

    return f"{override_prefix}No in-scope disruption notes found for this corridor and date window. Flagged for review."


class AnomalyVerifier:
    """Verifies flagged route-week anomalies using hybrid RAG and deterministic guardrails."""

    def __init__(
        self,
        vector_store: NoteVectorStore,
        enriched_notes: dict[str, EnrichedContextNote],
        llm_client: BaseLLMClient,
    ) -> None:
        self.vector_store = vector_store
        self.enriched_notes = enriched_notes
        self.llm_client = llm_client

    def verify_candidate(self, m: RouteWeekMetrics) -> FinalOutputRow:
        """Runs RAG retrieval, LLM analysis, deterministic guardrails, and outputs FinalOutputRow."""
        settings = get_settings()
        retrieved_notes = self.vector_store.retrieve_candidates(
            m.route, m.week_of, top_k=settings.RETRIEVAL_TOP_K
        )
        notes_text = "\n\n".join(
            f"Note ID: {n.note_id}\nDate: {n.note_date}\nApplies To: {n.applies_to}\nText: {n.note}"
            for n in retrieved_notes
        )

        user_prompt = f"""Route: {m.route} (Length type: {m.route_type})
Week of: {m.week_of} (Monday)
Cost per tonne-km: {m.cost_per_tonne_km:.2f} INR
Vs Own 8-Week Trailing History: {f"{m.vs_own_pct:+.1f}%" if m.vs_own_pct is not None else "N/A"}
Vs Peer Routes This Week: {f"{m.vs_peer_pct:+.1f}%" if m.vs_peer_pct is not None else "N/A"}

Retrieved Context Notes (sorted by date proximity):
{notes_text}
"""

        try:
            res = self.llm_client.complete_json(
                VERIFIER_SYSTEM_PROMPT,
                user_prompt,
                context=f"verify_{m.route}_{m.week_of}",
            )
            raw_verdict = str(res.get("verdict", "unexplained")).lower()
            raw_note_id = res.get("matched_note_id")
        except (
            FreightWatcherError,
            ValueError,
            KeyError,
            TypeError,
            json.JSONDecodeError,
            TimeoutError,
        ) as e:
            logger.error(
                "LLM failure on %s %s: %s; degrading safely to unexplained",
                m.route,
                m.week_of,
                e,
            )
            raw_verdict = "unexplained"
            raw_note_id = None

        final_verdict, verified_note_id, guardrail_reason = verify_verdict_guardrails(
            route=m.route,
            week_of=m.week_of,
            vs_own_pct=m.vs_own_pct,
            candidate_cptk=m.cost_per_tonne_km,
            llm_verdict=raw_verdict,
            matched_note_id=raw_note_id,
            notes_by_id=self.enriched_notes,
            max_weeks_open_ended=settings.MAX_WEEKS_OPEN_ENDED_EFFECT,
            magnitude_tolerance_pct=settings.MAGNITUDE_TOLERANCE_PCT,
            vs_peer_pct=m.vs_peer_pct,
            max_all_routes_peer_gap_pct=settings.ALL_ROUTES_MAX_PEER_GAP_PCT,
        )

        gate_overridden = bool(raw_verdict == "justified" and final_verdict == "unexplained")
        override_msg = guardrail_reason if gate_overridden else None

        flagged_status = "No (justified)" if final_verdict == "justified" else "Yes"
        final_reason = build_deterministic_reason(
            m=m,
            final_verdict=final_verdict,
            verified_note_id=verified_note_id,
            raw_verdict=raw_verdict,
            raw_note_id=raw_note_id,
            guardrail_override_reason=override_msg,
            retrieved_notes=retrieved_notes,
            enriched_notes=self.enriched_notes,
        )

        return FinalOutputRow(
            route=m.route,
            week_of=m.week_of,
            cost_per_tonne_km=m.cost_per_tonne_km,
            vs_own_pct=m.vs_own_pct,
            vs_peer_pct=m.vs_peer_pct,
            flagged=flagged_status,
            matched_note_id=verified_note_id,
            reason=final_reason,
            raw_llm_verdict=raw_verdict,
            raw_llm_matched_note_id=raw_note_id,
            gate_overridden=gate_overridden,
            override_reason=override_msg,
        )
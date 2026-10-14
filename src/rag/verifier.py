# src/rag/verifier.py
import json

from src.exceptions import FreightWatcherError
from src.guardrails import verify_verdict_guardrails
from src.llm.client import BaseLLMClient
from src.logger import get_logger
from src.rag.vector_store import NoteVectorStore
from src.schemas import EnrichedContextNote, FinalOutputRow, RouteWeekMetrics

logger = get_logger(__name__)

VERIFIER_SYSTEM_PROMPT = """You are an expert shipping cost anomaly verifier.
Your task is to determine whether a flagged cost rise on a shipping route is JUSTIFIED by real-world context notes, or remains UNEXPLAINED.

CRITICAL RULES:
1. Every "justified" verdict MUST cite the exact note_id.
2. The cited note must apply to the exact route (or 'All Routes') and its time window.
3. If no note genuinely justifies the cost rise, or notes say costs were NOT affected/absorbed, verdict MUST be "unexplained" and matched_note_id MUST be null. Never invent reasons.
4. Output JSON strictly matching:
{
  "verdict": "justified" | "unexplained",
  "matched_note_id": "NXXX" or null,
  "reason": "Plain-English explanation stating verdict and citing notes."
}
"""


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
        # 1. Recall-oriented top-K note retrieval
        retrieved_notes = self.vector_store.retrieve_candidates(m.route, m.week_of, top_k=3)
        notes_text = "\n\n".join(
            f"Note ID: {n.note_id}\nDate: {n.note_date}\nApplies To: {n.applies_to}\nText: {n.note}"
            for n in retrieved_notes
        )

        user_prompt = f"""Route: {m.route} (Length type: {m.route_type})
Week of: {m.week_of} (Monday)
Cost per tonne-km: {m.cost_per_tonne_km:.2f} INR
Vs Own 8-Week Trailing History: {f"{m.vs_own_pct:+.1f}%" if m.vs_own_pct is not None else "N/A"}
Vs Peer Routes This Week: {f"{m.vs_peer_pct:+.1f}%" if m.vs_peer_pct is not None else "N/A"}

Retrieved Context Notes:
{notes_text}
"""

        # 2. Call LLM (or fallback gracefully to unexplained on error)
        try:
            res = self.llm_client.complete_json(
                VERIFIER_SYSTEM_PROMPT,
                user_prompt,
                context=f"verify_{m.route}_{m.week_of}",
            )
            raw_verdict = str(res.get("verdict", "unexplained")).lower()
            raw_note_id = res.get("matched_note_id")
            raw_reason = str(res.get("reason", "Unexplained cost rise."))
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
            raw_reason = f"Automated verification degraded to unexplained due to provider error: {e}"

        # 3. Deterministic Guardrail Gate
        final_verdict, verified_note_id, guardrail_reason = verify_verdict_guardrails(
            route=m.route,
            week_of=m.week_of,
            vs_own_pct=m.vs_own_pct,
            candidate_cptk=m.cost_per_tonne_km,
            llm_verdict=raw_verdict,
            matched_note_id=raw_note_id,
            notes_by_id=self.enriched_notes,
        )

        # 4. Synthesize final reason
        flagged_status = "No (justified)" if final_verdict == "justified" else "Yes"
        if final_verdict == "unexplained" and raw_verdict == "justified":
            final_reason = f"{guardrail_reason} {raw_reason}"
        else:
            final_reason = raw_reason

        return FinalOutputRow(
            route=m.route,
            week_of=m.week_of,
            cost_per_tonne_km=m.cost_per_tonne_km,
            vs_own_pct=m.vs_own_pct,
            vs_peer_pct=m.vs_peer_pct,
            flagged=flagged_status,
            matched_note_id=verified_note_id,
            reason=final_reason,
        )
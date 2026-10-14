# src/agent/qa_agent.py
import json
from typing import Any

from src.exceptions import FreightWatcherError
from src.llm.client import BaseLLMClient
from src.logger import get_logger
from src.rag.vector_store import NoteVectorStore
from src.schemas import FinalOutputRow, RouteWeekMetrics

logger = get_logger(__name__)


class LogisticsCostAgent:
    """
    Agentic AI Assistant with multi-step tool-use and planning over pipeline results.
    Answers natural language queries about route pricing, historical trends,
    anomalies, and cited context notes.
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        evaluated_metrics: list[RouteWeekMetrics],
        flagged_rows: list[FinalOutputRow],
        enriched_notes: dict[str, Any],
        vector_store: NoteVectorStore,
    ) -> None:
        self.llm_client = llm_client
        self.metrics = evaluated_metrics
        self.flagged_rows = flagged_rows
        self.enriched_notes = enriched_notes
        self.vector_store = vector_store

        self._metrics_by_rw = {(m.route.lower(), m.week_of): m for m in self.metrics}
        self._flagged_by_rw = {(r.route.lower(), r.week_of): r for r in self.flagged_rows}

    def tool_get_weekly_metrics(self, route: str, week_of: str) -> dict:
        """Fetch exact cost, baseline comparison, and anomaly flag for a route and week."""
        key = (route.strip().lower(), week_of.strip())
        metric = self._metrics_by_rw.get(key)
        if not metric:
            return {"error": f"No data found for route '{route}' on week '{week_of}'."}

        flagged_row = self._flagged_by_rw.get(key)
        res = {
            "route": metric.route,
            "route_type": metric.route_type,
            "week_of": metric.week_of,
            "cost_per_tonne_km": round(metric.cost_per_tonne_km, 2),
            "vs_own_pct": round(metric.vs_own_pct, 1) if metric.vs_own_pct is not None else None,
            "vs_peer_pct": round(metric.vs_peer_pct, 1) if metric.vs_peer_pct is not None else None,
            "is_anomaly": metric.is_anomaly_candidate,
        }
        if flagged_row:
            res["flagged_verdict"] = flagged_row.flagged
            res["matched_note_id"] = flagged_row.matched_note_id
            res["reason"] = flagged_row.reason
        return res

    def tool_list_anomalies_for_route(self, route: str) -> list[dict]:
        """List all anomalous weeks and their findings for a specific route."""
        r_clean = route.strip().lower()
        matches = [r for (r_name, w), r in self._flagged_by_rw.items() if r_clean in r_name]
        return [
            {
                "route": row.route,
                "week_of": row.week_of,
                "cost_per_tonne_km": f"{row.cost_per_tonne_km:.2f}",
                "verdict": row.flagged,
                "matched_note_id": row.matched_note_id,
                "reason": row.reason,
            }
            for row in sorted(matches, key=lambda x: x.week_of)
        ]

    def tool_search_context_notes(self, query: str, route: str = "") -> list[dict]:
        """Search context notes for external disruptions, weather, fuel, or surcharges."""
        retrieved = self.vector_store.retrieve_candidates(route or "All Routes", "2025-01-01", top_k=3)
        return [
            {
                "note_id": n.note_id,
                "date": str(n.note_date),
                "applies_to": n.applies_to,
                "note": n.note,
                "cost_effect": n.cost_effect,
            }
            for n in retrieved
        ]

    def ask(self, user_query: str) -> str:
        """Plans tool usage based on user intent, gathers evidence, and formulates a plain-English answer."""
        q = user_query.lower()
        findings: list[str] = []

        target_route = None
        for r_name in {m.route for m in self.metrics}:
            if r_name.lower() in q:
                target_route = r_name
                break

        if target_route:
            anomalies = self.tool_list_anomalies_for_route(target_route)
            findings.append(
                f"Tool `list_anomalies_for_route({target_route})` returned "
                f"{len(anomalies)} flagged weeks: {json.dumps(anomalies)}"
            )

        notes_found = self.tool_search_context_notes(user_query, target_route or "")
        findings.append(f"Tool `search_context_notes` returned: {json.dumps(notes_found)}")

        system_prompt = (
            "You are a Senior Logistics Analyst AI. Formulate a direct, accurate, plain-English "
            "answer to the user's question using only the verified findings from the tools. "
            "Do not hallucinate reasons not provided in the findings."
        )
        user_prompt = f"User Question: {user_query}\n\nGathered Findings:\n" + "\n".join(findings)

        try:
            res = self.llm_client.complete_json(
                system_prompt,
                user_prompt + "\nOutput JSON with key 'answer': string",
                context="agent_qa",
            )
            if isinstance(res, dict):
                if "answer" in res:
                    return str(res["answer"])
                if "reason" in res:
                    verdict_prefix = f"Verdict: {res.get('verdict', '').title()}. " if res.get("verdict") else ""
                    return f"{verdict_prefix}{res['reason']}"
            return str(res)
        except (FreightWatcherError, ValueError, KeyError, TypeError) as e:
            logger.warning("Agent LLM completion failed (%s); building deterministic summary", e)
            if target_route:
                anoms = self.tool_list_anomalies_for_route(target_route)
                if anoms:
                    reasons = [f"- Week {a['week_of']}: {a['reason']}" for a in anoms]
                    return (
                        f"Analysis for {target_route}:\n"
                        f"Found {len(anoms)} anomaly weeks:\n" + "\n".join(reasons)
                    )
            return (
                f"Retrieved relevant notes regarding your query: "
                f"{', '.join(n['note_id'] + ' (' + n['note'] + ')' for n in notes_found[:2])}"
            )
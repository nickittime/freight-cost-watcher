import json
from typing import Any

from src.exceptions import FreightWatcherError
from src.llm.client import BaseLLMClient
from src.logger import get_logger
from src.rag.vector_store import NoteVectorStore
from src.schemas import FinalOutputRow, RouteWeekMetrics

logger = get_logger(__name__)

TOOL_PLANNER_PROMPT = """You are a logistics tool-calling planner.
Choose which tools to call to answer the user's question accurately.
Available tools:
1. get_flagged_rows(route: str | null, start_date: str | null, end_date: str | null)
   - Lists flagged anomaly weeks, verdicts, matched_note_id, and reasons.
2. get_weekly_metrics(route: str, start_date: str | null, end_date: str | null)
   - Returns weekly cost_per_tonne_km, vs_own_pct, vs_peer_pct for a route and date window.
3. search_notes(route: str | null, reference_date: str | null, query: str)
   - Searches context notes for a route around reference_date (YYYY-MM-DD).

Return JSON strictly in this format:
{
  "tool_calls": [
    {"name": "get_flagged_rows", "arguments": {"route": "Chennai-Bangalore", "start_date": "2025-02-01", "end_date": "2025-03-31"}},
    {"name": "search_notes", "arguments": {"route": "Chennai-Bangalore", "reference_date": "2025-02-24", "query": "flooding"}}
  ]
}
"""


class LogisticsCostAgent:
    """Agentic AI Assistant where the LLM dynamically selects and invokes data/RAG tools."""

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

    def get_flagged_rows(
        self,
        route: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[dict]:
        """Filter flagged anomaly rows by route and date range."""
        rows = self.flagged_rows
        if route:
            r_low = route.strip().lower()
            rows = [r for r in rows if r_low in r.route.lower()]
        if start_date:
            rows = [r for r in rows if r.week_of >= start_date]
        if end_date:
            rows = [r for r in rows if r.week_of <= end_date]
        return [
            {
                "route": r.route,
                "week_of": r.week_of,
                "cost_per_tonne_km": round(r.cost_per_tonne_km, 2),
                "vs_own_pct": round(r.vs_own_pct, 1) if r.vs_own_pct is not None else None,
                "vs_peer_pct": round(r.vs_peer_pct, 1) if r.vs_peer_pct is not None else None,
                "flagged": r.flagged,
                "matched_note_id": r.matched_note_id,
                "reason": r.reason,
            }
            for r in sorted(rows, key=lambda x: (x.week_of, x.route))
        ]

    def get_weekly_metrics(
        self,
        route: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[dict]:
        """Fetch weekly cost and baseline metrics for a route over a date window."""
        r_low = route.strip().lower()
        rows = [m for m in self.metrics if r_low in m.route.lower()]
        if start_date:
            rows = [m for m in rows if m.week_of >= start_date]
        if end_date:
            rows = [m for m in rows if m.week_of <= end_date]
        return [
            {
                "route": m.route,
                "week_of": m.week_of,
                "cost_per_tonne_km": round(m.cost_per_tonne_km, 2),
                "vs_own_pct": round(m.vs_own_pct, 1) if m.vs_own_pct is not None else None,
                "vs_peer_pct": round(m.vs_peer_pct, 1) if m.vs_peer_pct is not None else None,
                "is_anomaly_candidate": m.is_anomaly_candidate,
            }
            for m in sorted(rows, key=lambda x: x.week_of)[:20]
        ]

    def search_notes(
        self,
        route: str | None = None,
        reference_date: str | None = None,
        query: str = "",
    ) -> list[dict]:
        """Retrieve context notes scoped to route and ranked by proximity to reference_date."""
        ref_dt = reference_date or "2025-03-01"
        retrieved = self.vector_store.retrieve_candidates(
            route or "All Routes", ref_dt, top_k=5
        )
        return [
            {
                "note_id": n.note_id,
                "date": str(n.note_date),
                "applies_to": n.applies_to,
                "effective_start_date": str(n.effective_start_date),
                "effective_end_date": str(n.effective_end_date) if n.effective_end_date else None,
                "cost_effect": n.cost_effect,
                "note": n.note,
            }
            for n in retrieved
        ]

    def _execute_tool(self, name: str, args: dict) -> Any:
        if name == "get_flagged_rows":
            return self.get_flagged_rows(
                route=args.get("route"),
                start_date=args.get("start_date"),
                end_date=args.get("end_date"),
            )
        if name == "get_weekly_metrics":
            return self.get_weekly_metrics(
                route=args.get("route", ""),
                start_date=args.get("start_date"),
                end_date=args.get("end_date"),
            )
        if name == "search_notes":
            return self.search_notes(
                route=args.get("route"),
                reference_date=args.get("reference_date"),
                query=args.get("query", ""),
            )
        return {"error": f"Unknown tool: {name}"}

    def ask(self, user_query: str) -> str:
        """Plans tool calls via LLM, executes selected tools, and synthesizes a grounded answer."""
        try:
            plan = self.llm_client.complete_json(
                TOOL_PLANNER_PROMPT,
                f"User question: {user_query}",
                context="agent_plan",
            )
            tool_calls = plan.get("tool_calls", [])
        except (FreightWatcherError, ValueError, KeyError, TypeError):
            tool_calls = [{"name": "get_flagged_rows", "arguments": {}}]

        if not tool_calls:
            tool_calls = [{"name": "get_flagged_rows", "arguments": {}}]

        observations: list[dict] = []
        for call in tool_calls[:4]:
            t_name = call.get("name", "get_flagged_rows")
            t_args = call.get("arguments") or {}
            output = self._execute_tool(t_name, t_args)
            observations.append({"tool": t_name, "arguments": t_args, "output": output})

        synth_system = (
            "You are a Senior Logistics Analyst AI. Answer the user's question directly in plain "
            "English using ONLY the tool execution results. Cite exact week_of dates, "
            "cost_per_tonne_km numbers, and note IDs. Return JSON: {\"answer\": \"...\"}"
        )
        synth_user = (
            f"User Question: {user_query}\n\n"
            f"Tool Observations:\n{json.dumps(observations, indent=2)}"
        )

        try:
            res = self.llm_client.complete_json(
                synth_system, synth_user, context="agent_synthesize"
            )
            if isinstance(res, dict) and "answer" in res:
                return str(res["answer"])
        except (FreightWatcherError, ValueError, KeyError, TypeError) as e:
            logger.warning("Agent synthesis fallback triggered: %s", e)

        return json.dumps(observations, indent=2)
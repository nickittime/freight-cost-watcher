from datetime import date as Date
from typing import Literal

from pydantic import BaseModel, model_validator


def format_pct(val: float) -> str:
    """Signed, one decimal, never '-0.0%'. Zero renders as '+0.0%'."""
    rounded = round(val, 1)
    if rounded == 0:
        rounded = 0.0
    return f"{rounded:+.1f}%"


class EnrichedContextNote(BaseModel):
    """Note plus structured fields extracted once at index time; the guardrail gate reads these."""

    note_id: str
    note_date: Date
    applies_to: str
    note: str
    effective_start_date: Date
    effective_end_date: Date | None = None
    cost_effect: Literal["increase", "decrease", "none"]
    stated_magnitude_min_pct: float = 0.0
    stated_magnitude_max_pct: float = 0.0

    @model_validator(mode="after")
    def _check(self) -> "EnrichedContextNote":
        if self.effective_end_date and self.effective_end_date < self.effective_start_date:
            raise ValueError("effective_end_date precedes effective_start_date")
        if self.stated_magnitude_min_pct > self.stated_magnitude_max_pct:
            raise ValueError("magnitude min exceeds max")
        if self.cost_effect != "increase" and (
            self.stated_magnitude_min_pct or self.stated_magnitude_max_pct
        ):
            raise ValueError("magnitude must be 0 unless cost_effect == 'increase'")
        return self


class RouteWeekMetrics(BaseModel):
    route: str
    route_type: Literal["Short", "Medium", "Long"]
    week_of: str
    cost_per_tonne_km: float
    prior_weeks_count: int
    own_history_baseline: float | None
    peer_baseline: float | None
    vs_own_pct: float | None
    vs_peer_pct: float | None
    is_anomaly_candidate: bool = False


class FinalOutputRow(BaseModel):
    route: str
    week_of: str
    cost_per_tonne_km: float
    vs_own_pct: float | None
    vs_peer_pct: float | None
    flagged: Literal["Yes", "No (justified)"]
    matched_note_id: str | None = None
    reason: str

    @model_validator(mode="after")
    def _contract(self) -> "FinalOutputRow":
        if self.flagged == "Yes" and self.matched_note_id:
            raise ValueError("unexplained rows must have an empty matched_note_id")
        if self.flagged == "No (justified)" and not self.matched_note_id:
            raise ValueError("justified rows must cite a matched_note_id")
        return self

    def to_csv_dict(self) -> dict[str, str]:
        own = (
            "N/A (no prior history)"
            if self.vs_own_pct is None
            else f"{format_pct(self.vs_own_pct)} vs this route's past average"
        )
        peer = (
            "N/A (no peer routes)"
            if self.vs_peer_pct is None
            else f"{format_pct(self.vs_peer_pct)} vs similar-length routes this week"
        )
        prefix = "Verdict: Justified" if self.flagged == "No (justified)" else "Verdict: Unexplained"
        reason = self.reason.strip()
        if not reason.startswith("Verdict:"):
            reason = f"{prefix}. {reason}"
        return {
            "route": self.route,
            "week_of": self.week_of,
            "cost_per_tonne_km": f"{self.cost_per_tonne_km:.2f}",
            "vs_own_history": own,
            "vs_similar_routes": peer,
            "flagged": self.flagged,
            "matched_note_id": self.matched_note_id or "",
            "reason": reason,
        }
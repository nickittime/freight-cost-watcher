# src/schemas.py
from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


def format_pct(val: float | None) -> str:
    """Formats percentage change with signed display, e.g. +35.5% or N/A."""
    if val is None:
        return "N/A"
    rounded = round(val, 1)
    if rounded == 0.0:
        rounded = 0.0
    return f"{rounded:+.1f}%"


class RouteWeekMetrics(BaseModel):
    """Weekly aggregated unit economics and baseline deviations for a route."""

    model_config = ConfigDict(frozen=True)

    route: str
    route_type: Literal["Short", "Medium", "Long"]
    week_of: str
    cost_per_tonne_km: float
    prior_weeks_count: int
    own_history_baseline: float | None = None
    peer_baseline: float | None = None
    vs_own_pct: float | None = None
    vs_peer_pct: float | None = None
    is_anomaly_candidate: bool = False


class EnrichedContextNote(BaseModel):
    """Structured disruption note with normalized temporal bounds and causal metadata."""

    model_config = ConfigDict(frozen=True)

    note_id: str
    note_date: date
    applies_to: str
    note: str
    effective_start_date: date
    effective_end_date: date | None = None
    cost_effect: Literal["increase", "decrease", "none"] = "none"
    stated_magnitude_min_pct: float = 0.0
    stated_magnitude_max_pct: float = 0.0

    @model_validator(mode="after")
    def validate_temporal_window(self) -> "EnrichedContextNote":
        if self.effective_end_date is not None and self.effective_end_date < self.effective_start_date:
            raise ValueError(
                f"effective_end_date ({self.effective_end_date}) cannot be earlier than "
                f"effective_start_date ({self.effective_start_date})"
            )
        return self


class FinalOutputRow(BaseModel):
    """Clean public schema for the primary submission CSV."""

    model_config = ConfigDict(frozen=True)

    route: str = Field(..., description="Corridor name e.g. Mumbai-Pune")
    week_of: str = Field(..., description="Monday date string YYYY-MM-DD")
    cost_per_tonne_km: float = Field(..., description="Unrounded weekly unit cost")
    vs_own_pct: float | None = Field(default=None)
    vs_peer_pct: float | None = Field(default=None)
    flagged: str = Field(..., description="'Yes' or 'No (justified)'")
    matched_note_id: str | None = Field(default=None)
    reason: str = Field(...)

    # Audit fields for transparency and reproducibility checks
    raw_llm_verdict: str = Field(default="unexplained")
    raw_llm_matched_note_id: str | None = Field(default=None)
    gate_overridden: bool = Field(default=False)
    override_reason: str | None = Field(default=None)

    @model_validator(mode="after")
    def validate_output_contract(self) -> "FinalOutputRow":
        has_note = bool(self.matched_note_id and self.matched_note_id.strip())
        if self.flagged == "No (justified)" and not has_note:
            raise ValueError("Justified rows must cite a valid matched_note_id")
        if self.flagged == "Yes" and has_note:
            raise ValueError("Unexplained rows must not have a matched_note_id")
        return self

    def to_csv_dict(self) -> dict:
        own_str = (
            "N/A (no prior history)"
            if self.vs_own_pct is None
            else f"{format_pct(self.vs_own_pct)} vs this route's past average"
        )
        peer_str = (
            "N/A (no peer routes)"
            if self.vs_peer_pct is None
            else f"{format_pct(self.vs_peer_pct)} vs similar-length routes this week"
        )

        formatted_reason = self.reason
        if self.flagged == "No (justified)":
            if not formatted_reason.startswith("Verdict: Justified"):
                formatted_reason = f"Verdict: Justified - {formatted_reason}"
        else:
            if not formatted_reason.startswith("Verdict: Unexplained"):
                formatted_reason = f"Verdict: Unexplained - {formatted_reason}"

        return {
            "route": self.route,
            "week_of": self.week_of,
            "cost_per_tonne_km": f"{self.cost_per_tonne_km:.2f}",
            "vs_own_history": own_str,
            "vs_similar_routes": peer_str,
            "flagged": self.flagged,
            "matched_note_id": self.matched_note_id or "",
            "reason": formatted_reason,
        }

    def to_audit_dict(self) -> dict:
        return {
            "route": self.route,
            "week_of": self.week_of,
            "cost_per_tonne_km": f"{self.cost_per_tonne_km:.2f}",
            "raw_llm_verdict": self.raw_llm_verdict,
            "raw_llm_matched_note_id": self.raw_llm_matched_note_id or "",
            "final_verdict": self.flagged,
            "final_matched_note_id": self.matched_note_id or "",
            "gate_overridden": self.gate_overridden,
            "override_reason": self.override_reason or "",
        }
# src/guardrails.py
from datetime import date, datetime, timedelta

from src.logger import get_logger
from src.schemas import EnrichedContextNote

logger = get_logger(__name__)


def is_week_overlapping_note(
    week_monday: date,
    start_date: date,
    end_date: date | None,
    applies_to: str = "All Routes",
    max_weeks_open_ended: int = 8,
) -> bool:
    """Returns True if the Monday-to-Sunday week overlaps the note's effective window."""
    week_sunday = week_monday + timedelta(days=6)
    if end_date is not None:
        effective_end = end_date
    elif applies_to != "All Routes":
        effective_end = start_date + timedelta(days=6)
    else:
        effective_end = start_date + timedelta(weeks=max_weeks_open_ended)
    return max(week_monday, start_date) <= min(week_sunday, effective_end)


def diagnose_note_failure(
    note: EnrichedContextNote,
    route: str,
    week_monday: date,
    vs_own_pct: float | None,
    vs_peer_pct: float | None,
    max_weeks_open_ended: int = 8,
    magnitude_tolerance_pct: float = 3.0,
    max_all_routes_peer_gap_pct: float = 10.0,
) -> str | None:
    """
    Returns None if the note genuinely justifies this corridor and week.
    Otherwise returns a diagnostic reason string. Contains zero hard-coded route or date literals.
    """
    # 1. Scope check
    if note.applies_to != "All Routes" and note.applies_to != route:
        return f"applies to {note.applies_to}, but this route is {route}"

    # 2. Causal cost effect check
    if note.cost_effect != "increase":
        return f"reports no cost increase ('{note.cost_effect}'), which cannot justify an increase"

    # 3. Temporal overlap check
    week_sunday = week_monday + timedelta(days=6)
    if note.effective_end_date is not None:
        effective_end = note.effective_end_date
    elif note.applies_to != "All Routes":
        effective_end = note.effective_start_date + timedelta(days=6)
    else:
        effective_end = note.effective_start_date + timedelta(weeks=max_weeks_open_ended)

    if not (max(week_monday, note.effective_start_date) <= min(week_sunday, effective_end)):
        if week_monday > effective_end:
            return (
                f"falls outside the effective window (effective window closed on {effective_end} "
                f"prior to week {week_monday})"
            )
        return (
            f"falls outside the effective window (effective window begins on {note.effective_start_date} "
            f"after week {week_monday})"
        )

    # 4. Nationwide peer gap check
    if (
        note.applies_to == "All Routes"
        and vs_peer_pct is not None
        and vs_peer_pct > max_all_routes_peer_gap_pct
    ):
        return (
            f"nationwide note ('All Routes') affects peer routes equally and cannot explain "
            f"a +{vs_peer_pct:.1f}% divergence vs similar-length peers"
        )

    # 5. Stated magnitude check
    if note.stated_magnitude_max_pct > 0.0 and vs_own_pct is not None:
        max_allowed = note.stated_magnitude_max_pct + magnitude_tolerance_pct
        if vs_own_pct > max_allowed:
            return (
                f"observed cost rise (+{vs_own_pct:.1f}%) significantly exceeds stated impact "
                f"(+{note.stated_magnitude_max_pct:.1f}%, ceiling +{max_allowed:.1f}%)"
            )

    return None


def verify_verdict_guardrails(
    route: str,
    week_of: str,
    vs_own_pct: float | None,
    candidate_cptk: float,
    llm_verdict: str,
    matched_note_id: str | None,
    notes_by_id: dict[str, EnrichedContextNote],
    max_weeks_open_ended: int = 8,
    magnitude_tolerance_pct: float = 3.0,
    vs_peer_pct: float | None = None,
    max_all_routes_peer_gap_pct: float = 10.0,
) -> tuple[str, str | None, str]:
    """
    Validates proposed LLM justification against diagnostic guardrail checks.
    Returns (final_verdict, verified_note_id, guardrail_reason).
    """
    if llm_verdict != "justified" or not matched_note_id:
        return "unexplained", None, "No note cited by LLM."

    nid = matched_note_id.strip().upper()
    if nid not in notes_by_id:
        return "unexplained", None, f"Overridden: cited note '{nid}' does not exist in dataset."

    note = notes_by_id[nid]
    monday = datetime.strptime(week_of, "%Y-%m-%d").date()
    failure_diag = diagnose_note_failure(
        note=note,
        route=route,
        week_monday=monday,
        vs_own_pct=vs_own_pct,
        vs_peer_pct=vs_peer_pct,
        max_weeks_open_ended=max_weeks_open_ended,
        magnitude_tolerance_pct=magnitude_tolerance_pct,
        max_all_routes_peer_gap_pct=max_all_routes_peer_gap_pct,
    )

    if failure_diag is not None:
        logger.warning("Guardrail rejected note %s on %s %s: %s", nid, route, week_of, failure_diag)
        return "unexplained", None, f"Overridden: note {nid} {failure_diag}."

    return "justified", nid, f"Matches verified note {nid}."
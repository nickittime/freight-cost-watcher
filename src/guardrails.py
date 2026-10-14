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
    """
    Returns True if the Monday-to-Sunday week overlaps the note's effective window.
    - Explicit end_date: overlaps [start_date, end_date].
    - Route-specific note with end_date=None: bounded to 1 week [start_date, start_date + 6 days].
    - 'All Routes' note with end_date=None: bounded to max_weeks_open_ended weeks.
    """
    week_sunday = week_monday + timedelta(days=6)
    if end_date is not None:
        effective_end = end_date
    elif applies_to != "All Routes":
        effective_end = start_date + timedelta(days=6)
    else:
        effective_end = start_date + timedelta(weeks=max_weeks_open_ended)
    return max(week_monday, start_date) <= min(week_sunday, effective_end)


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
    Pure deterministic validation gate.

    Returns:
        (final_verdict, final_matched_note_id, deterministic_reason)
    """
    if llm_verdict != "justified" or not matched_note_id:
        return (
            "unexplained",
            None,
            "No matching disruption note justifies this cost rise; flagged for human review.",
        )

    note_id = matched_note_id.strip().upper()
    if note_id not in notes_by_id:
        logger.warning("Guardrail rejected non-existent note citation: %s", note_id)
        return (
            "unexplained",
            None,
            f"Overridden: cited note '{note_id}' does not exist in dataset.",
        )

    note = notes_by_id[note_id]

    # 1. Route Scope Check
    if note.applies_to != "All Routes" and note.applies_to != route:
        logger.warning(
            "Guardrail rejected route mismatch: %s applies to %s, not %s",
            note_id,
            note.applies_to,
            route,
        )
        return (
            "unexplained",
            None,
            f"Overridden: note {note_id} applies to {note.applies_to}, but this route is {route}.",
        )

    # 2. Temporal Overlap Check
    monday = datetime.strptime(week_of, "%Y-%m-%d").date()
    if not is_week_overlapping_note(
        monday,
        note.effective_start_date,
        note.effective_end_date,
        applies_to=note.applies_to,
        max_weeks_open_ended=max_weeks_open_ended,
    ):
        logger.warning(
            "Guardrail rejected temporal window mismatch for %s on week %s",
            note_id,
            week_of,
        )
        return (
            "unexplained",
            None,
            f"Overridden: week {week_of} falls outside the effective window of note {note_id}.",
        )

    # 3. Cost Effect Check
    if note.cost_effect != "increase":
        logger.warning(
            "Guardrail rejected non-increasing note %s (effect: %s)",
            note_id,
            note.cost_effect,
        )
        msg = (
            f"Overridden: note {note_id} describes '{note.cost_effect}' in costs, "
            "which cannot justify an increase."
        )
        return ("unexplained", None, msg)

    # 4. Nationwide ('All Routes') Peer Divergence Check
    if (
        note.applies_to == "All Routes"
        and vs_peer_pct is not None
        and vs_peer_pct > max_all_routes_peer_gap_pct
    ):
        logger.warning(
            "Guardrail rejected All Routes note %s for large peer gap (+%.1f%% > +%.1f%%)",
            note_id,
            vs_peer_pct,
            max_all_routes_peer_gap_pct,
        )
        msg = (
            f"Overridden: nationwide note {note_id} ('All Routes') affects peer routes equally "
            f"and cannot explain a +{vs_peer_pct:.1f}% divergence vs similar-length routes."
        )
        return ("unexplained", None, msg)

    # 5. Magnitude Check
    if note.stated_magnitude_max_pct > 0.0 and vs_own_pct is not None:
        max_allowed = note.stated_magnitude_max_pct + magnitude_tolerance_pct
        if vs_own_pct > max_allowed:
            logger.warning(
                "Guardrail rejected magnitude discrepancy: observed +%.1f%% exceeds "
                "stated +%.1f%% (+tol %.1f%%)",
                vs_own_pct,
                note.stated_magnitude_max_pct,
                max_allowed,
            )
            msg = (
                f"Overridden: observed cost rise (+{vs_own_pct:.1f}%) significantly exceeds the "
                f"impact described in note {note_id} (up to {note.stated_magnitude_max_pct:.1f}%)."
            )
            return ("unexplained", None, msg)

    return (
        "justified",
        note_id,
        f"Matches note {note_id} dated {note.note_date}: {note.note}",
    )
import inspect
from datetime import date

from src.guardrails import diagnose_note_failure
from src.rag.verifier import build_deterministic_reason
from src.schemas import EnrichedContextNote


def test_build_reason_contains_zero_route_or_week_literals():
    """Assert build_deterministic_reason source code contains no hard-coded route or date literals."""
    src = inspect.getsource(build_deterministic_reason)
    forbidden_literals = [
        "Chennai-Bangalore", "Ahmedabad-Mumbai", "Mumbai-Pune",
        "Delhi-Jaipur", "Mumbai-Delhi", "Kolkata-Bhubaneswar", "Delhi-Chennai",
        "2025-03-10", "2025-03-17", "2025-05-12", "2025-06-23", "2024-11-11", "2025-01-20",
    ]
    for lit in forbidden_literals:
        assert lit not in src, f"Found hard-coded literal '{lit}' in build_deterministic_reason"


def test_diagnose_note_out_of_window():
    n001 = EnrichedContextNote(
        note_id="N001",
        note_date=date(2025, 2, 24),
        applies_to="Chennai-Bangalore",
        note="Flooding from Feb 24 to Mar 8.",
        effective_start_date=date(2025, 2, 24),
        effective_end_date=date(2025, 3, 8),
        cost_effect="increase",
        stated_magnitude_min_pct=0.0,
        stated_magnitude_max_pct=100.0,
    )
    diag = diagnose_note_failure(
        note=n001,
        route="Chennai-Bangalore",
        week_monday=date(2025, 3, 10),
        vs_own_pct=22.1,
        vs_peer_pct=33.8,
    )
    assert diag is not None
    assert "effective window closed on 2025-03-08 prior to week 2025-03-10" in diag


def test_diagnose_note_all_routes_peer_gap():
    n003 = EnrichedContextNote(
        note_id="N003",
        note_date=date(2025, 5, 5),
        applies_to="All Routes",
        note="Diesel prices rose nationwide by 5-7%.",
        effective_start_date=date(2025, 5, 5),
        effective_end_date=None,
        cost_effect="increase",
        stated_magnitude_min_pct=5.0,
        stated_magnitude_max_pct=7.0,
    )
    diag = diagnose_note_failure(
        note=n003,
        route="Mumbai-Pune",
        week_monday=date(2025, 6, 23),
        vs_own_pct=6.2,
        vs_peer_pct=20.6,
        max_all_routes_peer_gap_pct=10.0,
    )
    assert diag is not None
    assert "cannot explain a +20.6% divergence" in diag


def test_diagnose_note_magnitude_exceeded():
    n003 = EnrichedContextNote(
        note_id="N003",
        note_date=date(2025, 5, 5),
        applies_to="All Routes",
        note="Diesel prices rose nationwide by 5-7%.",
        effective_start_date=date(2025, 5, 5),
        effective_end_date=None,
        cost_effect="increase",
        stated_magnitude_min_pct=5.0,
        stated_magnitude_max_pct=7.0,
    )
    diag = diagnose_note_failure(
        note=n003,
        route="Mumbai-Delhi",
        week_monday=date(2025, 5, 12),
        vs_own_pct=11.6,
        vs_peer_pct=6.7,
        magnitude_tolerance_pct=3.0,
    )
    assert diag is not None
    assert "exceeds stated impact" in diag
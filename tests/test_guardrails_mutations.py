# tests/test_guardrails_mutations.py
from datetime import date

import pytest

from src.guardrails import verify_verdict_guardrails
from src.schemas import EnrichedContextNote, FinalOutputRow


@pytest.fixture
def genuine_n001_note():
    return EnrichedContextNote(
        note_id="N001",
        note_date=date(2025, 2, 24),
        applies_to="Chennai-Bangalore",
        note="Heavy flooding on Chennai-Bangalore highway disrupted truck movement from Feb 24 to Mar 8.",
        effective_start_date=date(2025, 2, 24),
        effective_end_date=date(2025, 3, 8),
        cost_effect="increase",
        stated_magnitude_min_pct=0.0,
        stated_magnitude_max_pct=100.0,
    )


def test_guardrail_accepts_genuine_case(genuine_n001_note):
    verdict, note_id, _reason = verify_verdict_guardrails(
        route="Chennai-Bangalore",
        week_of="2025-02-24",
        vs_own_pct=31.5,
        candidate_cptk=3.58,
        llm_verdict="justified",
        matched_note_id="N001",
        notes_by_id={"N001": genuine_n001_note},
    )
    assert verdict == "justified"
    assert note_id == "N001"


def test_mutation_wrong_route_flips_to_unexplained(genuine_n001_note):
    """Mutating route to Mumbai-Pune must flip verdict to unexplained."""
    verdict, note_id, reason = verify_verdict_guardrails(
        route="Mumbai-Pune",
        week_of="2025-02-24",
        vs_own_pct=31.5,
        candidate_cptk=3.58,
        llm_verdict="justified",
        matched_note_id="N001",
        notes_by_id={"N001": genuine_n001_note},
    )
    assert verdict == "unexplained"
    assert note_id is None
    assert "applies to Chennai-Bangalore, but this route is Mumbai-Pune" in reason


def test_mutation_shifted_date_window_flips_to_unexplained(genuine_n001_note):
    """Week 2025-03-17 falls after flooding ended on Mar 8; must flip to unexplained."""
    verdict, note_id, reason = verify_verdict_guardrails(
        route="Chennai-Bangalore",
        week_of="2025-03-17",
        vs_own_pct=13.7,
        candidate_cptk=3.47,
        llm_verdict="justified",
        matched_note_id="N001",
        notes_by_id={"N001": genuine_n001_note},
    )
    assert verdict == "unexplained"
    assert note_id is None
    assert "falls outside the effective window" in reason


def test_mutation_negated_effect_flips_to_unexplained(genuine_n001_note):
    """Rewriting note to cost_effect='none' must be rejected."""
    mutated_note = genuine_n001_note.model_copy(
        update={
            "cost_effect": "none",
            "stated_magnitude_min_pct": 0.0,
            "stated_magnitude_max_pct": 0.0,
        }
    )
    verdict, note_id, reason = verify_verdict_guardrails(
        route="Chennai-Bangalore",
        week_of="2025-02-24",
        vs_own_pct=31.5,
        candidate_cptk=3.58,
        llm_verdict="justified",
        matched_note_id="N001",
        notes_by_id={"N001": mutated_note},
    )
    assert verdict == "unexplained"
    assert note_id is None
    assert "cannot justify an increase" in reason


def test_mutation_exceeded_magnitude_flips_to_unexplained():
    """Observed +25.0% vs stated max 7.0% (+ 3% tolerance = 10%) must be rejected."""
    fuel_note = EnrichedContextNote(
        note_id="N003",
        note_date=date(2025, 5, 5),
        applies_to="All Routes",
        note="Diesel prices rose nationwide, pushing up costs by roughly 5-7%.",
        effective_start_date=date(2025, 5, 5),
        effective_end_date=None,
        cost_effect="increase",
        stated_magnitude_min_pct=5.0,
        stated_magnitude_max_pct=7.0,
    )
    verdict, note_id, reason = verify_verdict_guardrails(
        route="Mumbai-Pune",
        week_of="2025-05-05",
        vs_own_pct=25.0,
        candidate_cptk=4.20,
        llm_verdict="justified",
        matched_note_id="N003",
        notes_by_id={"N003": fuel_note},
        magnitude_tolerance_pct=3.0,
    )
    assert verdict == "unexplained"
    assert note_id is None
    assert "significantly exceeds" in reason


def test_citation_consistency_output_contract():
    """Verify output contract rules: justified rows must cite notes; unexplained must not."""
    # Justified contract
    justified = FinalOutputRow(
        route="Ahmedabad-Mumbai",
        week_of="2025-01-20",
        cost_per_tonne_km=3.29,
        vs_own_pct=29.5,
        vs_peer_pct=22.5,
        flagged="No (justified)",
        matched_note_id="N002",
        reason="Matches note N002 dated 2025-01-20: festival surcharge.",
    ).to_csv_dict()
    assert justified["flagged"] == "No (justified)"
    assert justified["matched_note_id"] == "N002"
    assert "N002" in justified["reason"]
    assert "Verdict: Justified" in justified["reason"]

    # Unexplained contract
    unexplained = FinalOutputRow(
        route="Delhi-Jaipur",
        week_of="2024-11-11",
        cost_per_tonne_km=4.17,
        vs_own_pct=35.5,
        vs_peer_pct=21.0,
        flagged="Yes",
        matched_note_id="",
        reason="No matching note found.",
    ).to_csv_dict()
    assert unexplained["flagged"] == "Yes"
    assert unexplained["matched_note_id"] == ""
    assert "Verdict: Unexplained" in unexplained["reason"]


def test_mutation_all_routes_note_rejected_when_peer_gap_is_large():
    """Nationwide note N003 ('All Routes') cannot explain a +20.6% divergence vs peers."""
    fuel_note = EnrichedContextNote(
        note_id="N003",
        note_date=date(2025, 5, 5),
        applies_to="All Routes",
        note="Diesel prices rose nationwide starting this week, pushing up costs by 5-7%.",
        effective_start_date=date(2025, 5, 5),
        effective_end_date=None,
        cost_effect="increase",
        stated_magnitude_min_pct=5.0,
        stated_magnitude_max_pct=7.0,
    )
    verdict, note_id, reason = verify_verdict_guardrails(
        route="Mumbai-Pune",
        week_of="2025-06-23",
        vs_own_pct=6.2,
        candidate_cptk=3.76,
        llm_verdict="justified",
        matched_note_id="N003",
        notes_by_id={"N003": fuel_note},
        vs_peer_pct=20.6,
        max_all_routes_peer_gap_pct=10.0,
    )
    assert verdict == "unexplained"
    assert note_id is None
    assert "cannot explain a +20.6% divergence" in reason


def test_mutation_route_specific_null_end_date_defaults_to_one_week():
    """If N002 has effective_end_date=None, it only covers 1 week, not 7 weeks later."""
    n002_open = EnrichedContextNote(
        note_id="N002",
        note_date=date(2025, 1, 20),
        applies_to="Ahmedabad-Mumbai",
        note="Temporary festival surcharge on Ahmedabad-Mumbai.",
        effective_start_date=date(2025, 1, 20),
        effective_end_date=None,
        cost_effect="increase",
        stated_magnitude_min_pct=0.0,
        stated_magnitude_max_pct=100.0,
    )
    verdict, note_id, reason = verify_verdict_guardrails(
        route="Ahmedabad-Mumbai",
        week_of="2025-03-10",
        vs_own_pct=20.0,
        candidate_cptk=3.20,
        llm_verdict="justified",
        matched_note_id="N002",
        notes_by_id={"N002": n002_open},
        vs_peer_pct=22.0,
    )
    assert verdict == "unexplained"
    assert note_id is None
    assert "falls outside the effective window" in reason
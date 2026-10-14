import pytest
from pydantic import ValidationError

from src.schemas import EnrichedContextNote, FinalOutputRow, format_pct


def test_format_pct_signs_and_negative_zero():
    assert format_pct(35.52) == "+35.5%"
    assert format_pct(-12.34) == "-12.3%"
    assert format_pct(-0.04) == "+0.0%"
    assert format_pct(0.0) == "+0.0%"


def test_sample_row_strings():
    row = FinalOutputRow(
        route="Delhi-Jaipur", week_of="2024-11-11", cost_per_tonne_km=4.173,
        vs_own_pct=35.52, vs_peer_pct=21.03, flagged="Yes", reason="No matching note found.",
    ).to_csv_dict()
    assert row["cost_per_tonne_km"] == "4.17"
    assert row["vs_own_history"] == "+35.5% vs this route's past average"
    assert row["vs_similar_routes"] == "+21.0% vs similar-length routes this week"
    assert row["matched_note_id"] == ""
    assert row["reason"].startswith("Verdict: Unexplained")


def test_missing_history_text():
    row = FinalOutputRow(
        route="Mumbai-Delhi", week_of="2024-01-01", cost_per_tonne_km=2.04,
        vs_own_pct=None, vs_peer_pct=-0.04, flagged="Yes", reason="x",
    ).to_csv_dict()
    assert row["vs_own_history"] == "N/A (no prior history)"
    assert row["vs_similar_routes"].startswith("+0.0%")


def test_contract_blocks_uncited_justification():
    with pytest.raises(ValidationError):
        FinalOutputRow(route="A-B", week_of="2025-01-20", cost_per_tonne_km=3.0,
                       vs_own_pct=1.0, vs_peer_pct=1.0, flagged="No (justified)", reason="x")


def test_contract_blocks_note_on_unexplained():
    with pytest.raises(ValidationError):
        FinalOutputRow(route="A-B", week_of="2025-01-20", cost_per_tonne_km=3.0,
                       vs_own_pct=1.0, vs_peer_pct=1.0, flagged="Yes",
                       matched_note_id="N001", reason="x")


def test_enriched_note_rejects_inverted_window():
    with pytest.raises(ValidationError):
        EnrichedContextNote(
            note_id="N001", note_date="2025-02-24", applies_to="Chennai-Bangalore", note="t",
            effective_start_date="2025-03-08", effective_end_date="2025-02-24",
            cost_effect="increase",
        )
# tests/test_anomaly_detector.py
import pytest

from src.config import get_settings
from src.pipeline.anomaly_detector import AnomalyDetector
from src.pipeline.baselines import BaselineEngine
from src.pipeline.ingestion import load_and_aggregate_shipments
from src.schemas import RouteWeekMetrics


@pytest.fixture(scope="module")
def computed_metrics():
    settings = get_settings()
    weekly_df = load_and_aggregate_shipments(settings.shipments_path)
    engine = BaselineEngine(history_weeks=8)
    return engine.compute_metrics_deque(weekly_df)


def test_anomaly_detection_candidate_count(computed_metrics):
    detector = AnomalyDetector()
    all_evaluated, candidates = detector.evaluate(computed_metrics)

    assert len(all_evaluated) == 728
    # With hysteresis on Mumbai-Pune creep and acute spikes, candidates are bounded to 22 rows
    assert len(candidates) == 22


def test_sample_rows_are_flagged_candidates(computed_metrics):
    detector = AnomalyDetector()
    _, candidates = detector.evaluate(computed_metrics)
    cand_lookup = {(c.route, c.week_of) for c in candidates}

    # Row 1: Delhi-Jaipur acute spike
    assert ("Delhi-Jaipur", "2024-11-11") in cand_lookup

    # Row 2: Ahmedabad-Mumbai festival surcharge
    assert ("Ahmedabad-Mumbai", "2025-01-20") in cand_lookup

    # Row 3: Mumbai-Pune creep sample
    assert ("Mumbai-Pune", "2025-09-15") in cand_lookup


def test_hysteresis_state_machine_logic():
    """Unit test verifying entry, sustain, and exit conditions of creep hysteresis."""
    det = AnomalyDetector(
        spike_threshold=10.0,
        creep_peer_threshold=20.0,
        min_own_rise=5.0,
        hysteresis_keep_peer=15.0,
    )

    series = [
        RouteWeekMetrics(
            route="TestRoute",
            route_type="Short",
            week_of="2024-01-01",
            cost_per_tonne_km=3.0,
            prior_weeks_count=0,
            own_history_baseline=None,
            peer_baseline=2.4,
            vs_own_pct=None,
            vs_peer_pct=25.0,
        ),
        # Week 2: Enters creep (vs_peer=22% >= 20%, vs_own=6% >= 5%)
        RouteWeekMetrics(
            route="TestRoute",
            route_type="Short",
            week_of="2024-01-08",
            cost_per_tonne_km=3.18,
            prior_weeks_count=1,
            own_history_baseline=3.0,
            peer_baseline=2.6,
            vs_own_pct=6.0,
            vs_peer_pct=22.0,
        ),
        # Week 3: Sustains creep via hysteresis (vs_peer=17% >= 15%, vs_own=2% > 0%)
        RouteWeekMetrics(
            route="TestRoute",
            route_type="Short",
            week_of="2024-01-15",
            cost_per_tonne_km=3.24,
            prior_weeks_count=2,
            own_history_baseline=3.09,
            peer_baseline=2.76,
            vs_own_pct=2.0,
            vs_peer_pct=17.0,
        ),
        # Week 4: Exits creep (vs_peer=13% < 15%)
        RouteWeekMetrics(
            route="TestRoute",
            route_type="Short",
            week_of="2024-01-22",
            cost_per_tonne_km=3.27,
            prior_weeks_count=3,
            own_history_baseline=3.14,
            peer_baseline=2.89,
            vs_own_pct=1.0,
            vs_peer_pct=13.0,
        ),
        # Week 5: Does not re-enter (vs_peer=17% is below the 20% entry threshold)
        RouteWeekMetrics(
            route="TestRoute",
            route_type="Short",
            week_of="2024-01-29",
            cost_per_tonne_km=3.30,
            prior_weeks_count=4,
            own_history_baseline=3.17,
            peer_baseline=2.82,
            vs_own_pct=2.0,
            vs_peer_pct=17.0,
        ),
    ]

    _, candidates = det.evaluate(series)
    flagged_weeks = [c.week_of for c in candidates]
    assert flagged_weeks == ["2024-01-08", "2024-01-15"]


def test_first_week_never_flags_on_unobserved_history():
    """First week of a route has vs_own_pct=None, so it should not flag as an acute spike."""
    det = AnomalyDetector(spike_threshold=10.0, creep_peer_threshold=20.0)
    m = RouteWeekMetrics(
        route="NewRoute",
        route_type="Long",
        week_of="2024-01-01",
        cost_per_tonne_km=5.0,
        prior_weeks_count=0,
        own_history_baseline=None,
        peer_baseline=5.0,
        vs_own_pct=None,
        vs_peer_pct=0.0,
    )
    _, candidates = det.evaluate([m])
    assert len(candidates) == 0


def test_sensitivity_analysis_stability(computed_metrics):
    detector = AnomalyDetector()
    sens_df = detector.run_sensitivity_analysis(computed_metrics)
    assert not sens_df.empty
    # Across all grid combinations, candidate count stays bounded between 18 and 24
    assert sens_df["candidate_count"].min() >= 18
    assert sens_df["candidate_count"].max() <= 24
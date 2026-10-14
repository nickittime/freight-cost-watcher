from datetime import date, timedelta

import pandas as pd
import pytest

from src.config import get_settings
from src.exceptions import DataValidationError
from src.pipeline.baselines import BaselineEngine
from src.pipeline.ingestion import (
    aggregate_weekly,
    load_and_aggregate_shipments,
    to_week_monday,
    validate_shipments,
)


@pytest.fixture(scope="module")
def weekly():
    return load_and_aggregate_shipments(get_settings().shipments_path)


def _weekly_frame(rows):
    return pd.DataFrame(rows, columns=["route", "route_type", "week_of", "cost_per_tonne_km"])


def _mondays(n, start=date(2024, 1, 1)):
    return [(start + timedelta(weeks=i)).isoformat() for i in range(n)]


# ---------- ingestion ----------
def test_week_monday_mapping():
    s = pd.Series(["2024-01-01", "2024-01-07", "2024-01-08"])  # Mon, Sun, Mon
    assert to_week_monday(s).tolist() == ["2024-01-01", "2024-01-01", "2024-01-08"]


def _shipments(**overrides):
    base = {
        "shipment_id": "S1",
        "origin": "A",
        "destination": "B",
        "route_type": "Short",
        "quantity_tonnes": 10.0,
        "distance_km": 100.0,
        "freight_cost_inr": 3000.0,
        "shipment_date": "2024-01-03",
    }
    base.update(overrides)
    return base


@pytest.mark.parametrize(
    "bad",
    [
        {"quantity_tonnes": 0},
        {"distance_km": -5},
        {"freight_cost_inr": None},
        {"route_type": "Tiny"},
        {"shipment_date": "03/01/2024"},
    ],
)
def test_validation_rejects_bad_rows(bad):
    with pytest.raises(DataValidationError):
        validate_shipments(pd.DataFrame([_shipments(**bad)]))


def test_validation_rejects_duplicate_ids_and_split_route_types():
    with pytest.raises(DataValidationError):
        validate_shipments(pd.DataFrame([_shipments(), _shipments()]))
    with pytest.raises(DataValidationError):
        validate_shipments(
            pd.DataFrame([_shipments(), _shipments(shipment_id="S2", route_type="Long")])
        )


def test_ratio_of_sums_not_mean_of_ratios():
    df = validate_shipments(
        pd.DataFrame([
            _shipments(
                shipment_id="S1",
                quantity_tonnes=10,
                distance_km=100,
                freight_cost_inr=1000,
            ),
            _shipments(
                shipment_id="S2",
                quantity_tonnes=30,
                distance_km=100,
                freight_cost_inr=6000,
            ),
        ])
    )
    got = aggregate_weekly(df)["cost_per_tonne_km"].iloc[0]
    assert got == pytest.approx(7000 / 4000)  # 1.75, whereas mean of ratios would be 1.5


def test_ingestion_totals(weekly):
    assert len(weekly) == 728 and weekly["cost_per_tonne_km"].min() > 0


# ---------- baselines ----------
def test_deque_matches_pandas_reference(weekly):
    got = pd.DataFrame([m.model_dump() for m in BaselineEngine().compute_metrics_deque(weekly)])
    got = got.sort_values(["route", "week_of"]).reset_index(drop=True)
    ref = BaselineEngine.compute_metrics_pandas(weekly)
    for col in ["own_history_baseline", "peer_baseline", "vs_own_pct", "vs_peer_pct"]:
        pd.testing.assert_series_equal(
            got[col].astype(float), ref[col].astype(float), check_names=False, rtol=1e-12
        )  # Also asserts NaN positions match
    assert got["prior_weeks_count"].tolist() == ref["prior_weeks_count"].tolist()


def test_sample_output_rows(weekly):
    res = {(m.route, m.week_of): m for m in BaselineEngine().compute_metrics_deque(weekly)}
    for key, cptk, own, peer in [
        (("Delhi-Jaipur", "2024-11-11"), "4.17", "+35.5%", "+21.0%"),
        (("Ahmedabad-Mumbai", "2025-01-20"), "3.29", "+29.5%", "+22.5%"),
        (("Mumbai-Pune", "2025-09-15"), "3.98", "+9.2%", "+23.6%"),
    ]:
        m = res[key]
        assert (
            f"{m.cost_per_tonne_km:.2f}",
            f"{m.vs_own_pct:+.1f}%",
            f"{m.vs_peer_pct:+.1f}%",
        ) == (cptk, own, peer)


def test_window_is_exactly_eight_prior_weeks():
    rows = [("R", "Short", w, float(i + 1)) for i, w in enumerate(_mondays(10))]
    m = BaselineEngine().compute_metrics_deque(_weekly_frame(rows))
    assert (m[0].prior_weeks_count, m[0].own_history_baseline) == (0, None)
    assert (m[2].prior_weeks_count, m[2].own_history_baseline) == (2, 1.5)  # Mean(1,2)
    assert (m[8].prior_weeks_count, m[8].own_history_baseline) == (8, 4.5)  # Mean(1..8)
    assert (m[9].prior_weeks_count, m[9].own_history_baseline) == (8, 5.5)  # Mean(2..9)


def test_gap_expires_history_but_boundary_is_inclusive():
    eng = BaselineEngine()
    exactly_8_weeks = _weekly_frame([
        ("R", "Short", "2024-01-01", 3.0),
        ("R", "Short", "2024-02-26", 6.0),
    ])
    assert eng.compute_metrics_deque(exactly_8_weeks)[1].prior_weeks_count == 1  # 56 days: kept
    ten_weeks = _weekly_frame([
        ("R", "Short", "2024-01-01", 3.0),
        ("R", "Short", "2024-03-11", 6.0),
    ])
    assert eng.compute_metrics_deque(ten_weeks)[1].prior_weeks_count == 0  # 70 days: expired


def test_peer_average_excludes_self_and_handles_singletons():
    w = "2024-01-01"
    m = {
        x.route: x
        for x in BaselineEngine().compute_metrics_deque(
            _weekly_frame([
                ("A", "Short", w, 2.0),
                ("B", "Short", w, 3.0),
                ("C", "Short", w, 4.0),
                ("D", "Long", w, 9.0),
            ])
        )
    }
    assert m["A"].peer_baseline == pytest.approx(3.5)
    assert m["C"].peer_baseline == pytest.approx(2.5)
    assert m["D"].peer_baseline is None and m["D"].vs_peer_pct is None


def test_no_look_ahead(weekly):
    cutoff = "2025-01-20"
    full = BaselineEngine().compute_metrics_deque(weekly)
    trunc = BaselineEngine().compute_metrics_deque(weekly[weekly["week_of"] <= cutoff])
    expected = [m.model_dump() for m in full if m.week_of <= cutoff]
    assert [m.model_dump() for m in trunc] == expected  # Future data cannot change past metrics


def test_out_of_order_weeks_rejected():
    eng = BaselineEngine()
    eng.observe_week("2024-01-08", _weekly_frame([("R", "Short", "2024-01-08", 3.0)]))
    with pytest.raises(DataValidationError):
        eng.observe_week("2024-01-01", _weekly_frame([("R", "Short", "2024-01-01", 3.0)]))
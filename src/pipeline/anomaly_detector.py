from collections import defaultdict

import pandas as pd

from src.config import get_settings
from src.logger import get_logger
from src.schemas import RouteWeekMetrics

logger = get_logger(__name__)


class AnomalyDetector:
    """
    Evaluates weekly route metrics and flags cost-rise anomaly candidates.

    Uses acute spike detection and a hysteresis state machine for sustained creeps.
    """

    def __init__(
        self,
        spike_threshold: float | None = None,
        creep_peer_threshold: float | None = None,
        min_own_rise: float | None = None,
        hysteresis_keep_peer: float | None = None,
    ) -> None:
        settings = get_settings()
        self.spike_threshold = (
            spike_threshold
            if spike_threshold is not None
            else settings.VS_OWN_SPIKE_THRESHOLD_PCT
        )
        self.creep_peer_threshold = (
            creep_peer_threshold
            if creep_peer_threshold is not None
            else settings.VS_PEER_CREEP_THRESHOLD_PCT
        )
        self.min_own_rise = (
            min_own_rise
            if min_own_rise is not None
            else settings.MIN_OWN_RISE_FOR_PEER_CREEP_PCT
        )
        self.hysteresis_keep_peer = (
            hysteresis_keep_peer
            if hysteresis_keep_peer is not None
            else settings.HYSTERESIS_KEEP_PEER_PCT
        )

    def evaluate(
        self, metrics: list[RouteWeekMetrics]
    ) -> tuple[list[RouteWeekMetrics], list[RouteWeekMetrics]]:
        """
        Evaluates metrics route-by-route in chronological order.

        Returns:
            (all_evaluated_metrics, candidate_anomalies_only)
        """
        metrics_by_route = defaultdict(list)
        for m in metrics:
            metrics_by_route[m.route].append(m)

        all_evaluated: list[RouteWeekMetrics] = []
        candidates: list[RouteWeekMetrics] = []

        for route in sorted(metrics_by_route.keys()):
            r_metrics = sorted(metrics_by_route[route], key=lambda x: x.week_of)
            in_creep_state = False

            for m in r_metrics:
                v_own = m.vs_own_pct
                v_peer = m.vs_peer_pct

                # 1. Acute Spike Trigger (requires observable prior history)
                is_spike = v_own is not None and v_own >= self.spike_threshold

                # 2. Creep Entry Trigger
                creep_entry = (
                    v_peer is not None
                    and v_peer >= self.creep_peer_threshold
                    and v_own is not None
                    and v_own >= self.min_own_rise
                )

                if creep_entry:
                    in_creep_state = True
                elif in_creep_state:
                    # 3. Hysteresis Sustain Condition
                    if (
                        v_peer is not None
                        and v_peer >= self.hysteresis_keep_peer
                        and v_own is not None
                        and v_own > 0.0
                    ):
                        in_creep_state = True
                    else:
                        in_creep_state = False

                is_anomaly = is_spike or in_creep_state

                # Return updated immutable model
                updated_m = m.model_copy(update={"is_anomaly_candidate": is_anomaly})
                all_evaluated.append(updated_m)
                if is_anomaly:
                    candidates.append(updated_m)

        logger.info(
            "Evaluated %d route-weeks: flagged %d candidate anomalies (%.1f%%).",
            len(all_evaluated),
            len(candidates),
            len(candidates) / len(all_evaluated) * 100 if all_evaluated else 0.0,
        )
        return all_evaluated, candidates

    def run_sensitivity_analysis(
        self,
        metrics: list[RouteWeekMetrics],
        spike_range: tuple[float, ...] = (8.0, 10.0, 12.0),
        creep_peer_range: tuple[float, ...] = (18.0, 20.0, 22.0),
        min_rise_range: tuple[float, ...] = (4.0, 5.0, 6.0),
        keep_peer_range: tuple[float, ...] = (12.0, 15.0),
    ) -> pd.DataFrame:
        """
        Runs a grid analysis across threshold combinations to verify stability.

        Outputs a DataFrame proving the candidate pool remains stable.
        """
        records = []
        for spike in spike_range:
            for creep_peer in creep_peer_range:
                for min_rise in min_rise_range:
                    for keep_peer in keep_peer_range:
                        det = AnomalyDetector(
                            spike_threshold=spike,
                            creep_peer_threshold=creep_peer,
                            min_own_rise=min_rise,
                            hysteresis_keep_peer=keep_peer,
                        )
                        _, cands = det.evaluate(metrics)
                        records.append({
                            "spike_threshold": spike,
                            "creep_peer_threshold": creep_peer,
                            "min_own_rise": min_rise,
                            "hysteresis_keep_peer": keep_peer,
                            "candidate_count": len(cands),
                        })
        return pd.DataFrame(records)
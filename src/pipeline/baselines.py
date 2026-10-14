from collections import deque
from datetime import date, datetime

import pandas as pd

from src.exceptions import DataValidationError
from src.schemas import RouteWeekMetrics


class BaselineEngine:
    """
    Trailing-window own-history baseline and leave-one-out peer baseline.

    Complexity per route-week:
    - own history: mean over a per-route deque of at most `history_weeks` entries,
      O(window) = O(8), constant in dataset size. Recomputed rather than kept as a
      running sum so results are bit-stable (no incremental float drift).
    - peer baseline: O(1) leave-one-out, (sum - self) / (n - 1).

    The window is calendar-based: entries older than history_weeks * 7 days before the
    current Monday are dropped, so a missing week never lets stale data pose as recent.
    State is incremental: feed weeks in chronological order via `observe_week`.
    """

    def __init__(self, history_weeks: int = 8) -> None:
        if history_weeks < 1:
            raise ValueError("history_weeks must be >= 1")
        self.history_weeks = history_weeks
        self._max_age_days = history_weeks * 7
        self._history: dict[str, deque[tuple[date, float]]] = {}
        self._last_week: date | None = None

    def observe_week(self, week_of: str, week_df: pd.DataFrame) -> list[RouteWeekMetrics]:
        """Compute metrics for one week's route rows, then add them to history."""
        monday = datetime.strptime(week_of, "%Y-%m-%d").date()
        if self._last_week is not None and monday <= self._last_week:
            raise DataValidationError(
                f"Weeks must be observed in strictly increasing order: {week_of} after {self._last_week}"
            )
        if week_df["route"].duplicated().any():
            raise DataValidationError(f"Duplicate route rows within week {week_of}")

        rows = week_df.sort_values("route")
        type_sum: dict[str, float] = {}
        type_cnt: dict[str, int] = {}
        for rtype, cptk in zip(rows["route_type"], rows["cost_per_tonne_km"]):
            type_sum[rtype] = type_sum.get(rtype, 0.0) + float(cptk)
            type_cnt[rtype] = type_cnt.get(rtype, 0) + 1

        results: list[RouteWeekMetrics] = []
        for route, rtype, cptk in zip(rows["route"], rows["route_type"], rows["cost_per_tonne_km"]):
            cptk = float(cptk)
            hist = self._history.setdefault(route, deque())
            while hist and (monday - hist[0][0]).days > self._max_age_days:
                hist.popleft()

            prior = len(hist)
            own_base = sum(v for _, v in hist) / prior if prior else None
            vs_own = (cptk - own_base) / own_base * 100.0 if own_base else None

            n = type_cnt[rtype]
            peer_base = (type_sum[rtype] - cptk) / (n - 1) if n > 1 else None
            vs_peer = (cptk - peer_base) / peer_base * 100.0 if peer_base else None

            results.append(
                RouteWeekMetrics(
                    route=route,
                    route_type=rtype,
                    week_of=week_of,
                    cost_per_tonne_km=cptk,
                    prior_weeks_count=prior,
                    own_history_baseline=own_base,
                    peer_baseline=peer_base,
                    vs_own_pct=vs_own,
                    vs_peer_pct=vs_peer,
                )
            )
            hist.append((monday, cptk))

        self._last_week = monday
        return results

    def compute_metrics_deque(self, weekly_df: pd.DataFrame) -> list[RouteWeekMetrics]:
        if weekly_df.empty:
            raise DataValidationError("Cannot compute baselines on an empty DataFrame")
        self._history, self._last_week = {}, None
        out: list[RouteWeekMetrics] = []
        for week_of, group in weekly_df.groupby("week_of", sort=True):
            out.extend(self.observe_week(week_of, group))
        return out

    @staticmethod
    def compute_metrics_pandas(weekly_df: pd.DataFrame, history_weeks: int = 8) -> pd.DataFrame:
        """Independent vectorized reference used only to cross-validate the deque engine."""
        wdf = weekly_df.sort_values(["route", "week_of"]).reset_index(drop=True)
        cptk = wdf["cost_per_tonne_km"]
        by_route = cptk.groupby(wdf["route"])
        wdf["own_history_baseline"] = by_route.transform(
            lambda s: s.shift(1).rolling(history_weeks, min_periods=1).mean()
        )
        wdf["prior_weeks_count"] = by_route.transform(
            lambda s: s.shift(1).rolling(history_weeks, min_periods=1).count()
        ).fillna(0).astype(int)

        grp = cptk.groupby([wdf["route_type"], wdf["week_of"]])
        n, total = grp.transform("count"), grp.transform("sum")
        wdf["peer_baseline"] = ((total - cptk) / (n - 1)).where(n > 1)
        wdf["vs_own_pct"] = (cptk - wdf["own_history_baseline"]) / wdf["own_history_baseline"] * 100.0
        wdf["vs_peer_pct"] = (cptk - wdf["peer_baseline"]) / wdf["peer_baseline"] * 100.0
        return wdf
from pathlib import Path

import pandas as pd

from src.exceptions import DataValidationError
from src.logger import get_logger

logger = get_logger(__name__)

REQUIRED_COLUMNS = (
    "shipment_id",
    "origin",
    "destination",
    "route_type",
    "quantity_tonnes",
    "distance_km",
    "freight_cost_inr",
    "shipment_date",
)
NUMERIC_COLUMNS = ("quantity_tonnes", "distance_km", "freight_cost_inr")
VALID_ROUTE_TYPES = {"Short", "Medium", "Long"}


def to_week_monday(dates: pd.Series) -> pd.Series:
    """Map ISO dates to the Monday (YYYY-MM-DD) of their Monday-Sunday week."""
    parsed = pd.to_datetime(dates, format="%Y-%m-%d", errors="coerce")
    if parsed.isna().any():
        bad = dates[parsed.isna()].head(5).tolist()
        raise DataValidationError(f"Unparseable shipment_date (expected YYYY-MM-DD), e.g. {bad}")
    monday = parsed - pd.to_timedelta(parsed.dt.weekday, unit="D")
    return monday.dt.strftime("%Y-%m-%d")


def validate_shipments(raw: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy with `route`, `week_of`, `tonne_km`; raise on any invalid data."""
    missing = set(REQUIRED_COLUMNS) - set(raw.columns)
    if missing:
        raise DataValidationError(f"Missing required columns: {sorted(missing)}")

    df = raw.copy()
    for col in ("origin", "destination", "route_type", "shipment_id"):
        df[col] = df[col].astype(str).str.strip()

    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        bad = df[df[col].isna() | (df[col] <= 0)]
        if not bad.empty:
            raise DataValidationError(
                f"{len(bad)} rows have missing/non-positive {col}, e.g. {bad['shipment_id'].head(3).tolist()}"
            )

    bad_types = set(df["route_type"]) - VALID_ROUTE_TYPES
    if bad_types:
        raise DataValidationError(f"Unknown route_type values: {sorted(bad_types)}")

    dupes = df.loc[df["shipment_id"].duplicated(), "shipment_id"]
    if not dupes.empty:
        raise DataValidationError(f"Duplicate shipment_id values, e.g. {dupes.head(3).tolist()}")

    df["route"] = df["origin"] + "-" + df["destination"]  # directional by design
    types_per_route = df.groupby("route")["route_type"].nunique()
    inconsistent = types_per_route[types_per_route > 1].index.tolist()
    if inconsistent:
        raise DataValidationError(f"Routes with more than one route_type: {inconsistent}")

    df["week_of"] = to_week_monday(df["shipment_date"].astype(str).str.strip())
    df["tonne_km"] = df["quantity_tonnes"] * df["distance_km"]
    return df


def aggregate_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """Weekly cost per tonne-km as a ratio of sums: sum(cost) / sum(tonnes * km)."""
    weekly = df.groupby(["route", "route_type", "week_of"], as_index=False).agg(
        total_cost=("freight_cost_inr", "sum"),
        total_tonne_km=("tonne_km", "sum"),
        shipment_count=("shipment_id", "count"),
    )
    weekly["cost_per_tonne_km"] = weekly["total_cost"] / weekly["total_tonne_km"]
    return weekly.sort_values(["week_of", "route"]).reset_index(drop=True)


def load_and_aggregate_shipments(file_path: Path) -> pd.DataFrame:
    if not file_path.exists():
        raise DataValidationError(f"Shipment records file not found: {file_path}")
    clean = validate_shipments(pd.read_csv(file_path))
    weekly = aggregate_weekly(clean)
    logger.info(
        "Loaded %d shipments -> %d route-weeks across %d routes",
        len(clean),
        len(weekly),
        weekly["route"].nunique(),
    )
    return weekly
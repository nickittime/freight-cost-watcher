# src/rag/note_enricher.py
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.config import get_settings
from src.llm.client import BaseLLMClient
from src.logger import get_logger
from src.schemas import EnrichedContextNote

logger = get_logger(__name__)


ENRICHMENT_SYSTEM_PROMPT = """You are a precision logistics domain analyst.
Extract structured metadata from the shipping disruption context note.
Return JSON with:
{
  "effective_start_date": "YYYY-MM-DD",
  "effective_end_date": "YYYY-MM-DD" or null,
  "cost_effect": "increase" | "decrease" | "none",
  "stated_magnitude_min_pct": float,
  "stated_magnitude_max_pct": float
}
"""


def load_or_enrich_notes(
    notes_csv_path: Path,
    llm_client: BaseLLMClient,
    use_cache: bool = True,
) -> dict[str, EnrichedContextNote]:
    """Loads and caches structured context notes for deterministic guardrail evaluation."""
    settings = get_settings()
    settings.ensure_dirs()
    cache_path = settings.CACHE_DIR / "enriched_notes.json"

    if use_cache and cache_path.exists():
        logger.info("Loading cached enriched notes from %s", cache_path)
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {nid: EnrichedContextNote(**props) for nid, props in data.items()}

    df_notes = pd.read_csv(notes_csv_path)
    enriched_dict: dict[str, EnrichedContextNote] = {}

    for _, row in df_notes.iterrows():
        nid = str(row["note_id"]).strip()
        ndate = datetime.strptime(str(row["date"]).strip(), "%Y-%m-%d").date()
        applies = str(row["applies_to"]).strip()
        text = str(row["note"]).strip()

        if settings.LLM_PROVIDER == "fake":
            res = llm_client.complete_json(ENRICHMENT_SYSTEM_PROMPT, f"Note {nid}: {text}")
        else:
            prompt = (
                f"Note ID: {nid}\nPublished Date: {ndate}\n"
                f"Applies To: {applies}\nText: {text}"
            )
            res = llm_client.complete_json(
                ENRICHMENT_SYSTEM_PROMPT, prompt, context=f"enrich_{nid}"
            )

        # Defensive field extraction
        start_str = (
            res.get("effective_start_date")
            or res.get("start_date")
            or str(ndate)
        )
        end_str = res.get("effective_end_date") or res.get("end_date")
        effect = res.get("cost_effect", "none")
        if effect not in ("increase", "decrease", "none"):
            effect = "none"

        min_mag = float(res.get("stated_magnitude_min_pct", 0.0) or 0.0)
        max_mag = float(res.get("stated_magnitude_max_pct", 0.0) or 0.0)

        # Guard against schema inversion (e.g. min > max)
        if min_mag > max_mag:
            min_mag, max_mag = max_mag, min_mag

        start_date = datetime.strptime(start_str, "%Y-%m-%d").date()
        end_date = (
            datetime.strptime(end_str, "%Y-%m-%d").date()
            if end_str and str(end_str).lower() not in ("none", "null")
            else None
        )

        enriched = EnrichedContextNote(
            note_id=nid,
            note_date=ndate,
            applies_to=applies,
            note=text,
            effective_start_date=start_date,
            effective_end_date=end_date,
            cost_effect=effect,
            stated_magnitude_min_pct=min_mag,
            stated_magnitude_max_pct=max_mag,
        )
        enriched_dict[nid] = enriched

    # Save to cache
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(
            {nid: note.model_dump(mode="json") for nid, note in enriched_dict.items()},
            f,
            indent=2,
        )

    logger.info("Enriched and cached %d context notes.", len(enriched_dict))
    return enriched_dict
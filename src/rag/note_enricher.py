import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.config import get_settings
from src.exceptions import FreightWatcherError
from src.llm.client import BaseLLMClient
from src.logger import get_logger
from src.schemas import EnrichedContextNote

logger = get_logger(__name__)


BATCH_ENRICHMENT_SYSTEM_PROMPT = """You are a precision logistics domain analyst.
Extract structured metadata for each shipping disruption context note provided.
Return a JSON object mapping each note_id (e.g. "N001") to its metadata object:
{
  "N001": {
    "effective_start_date": "YYYY-MM-DD",
    "effective_end_date": "YYYY-MM-DD" or null,
    "cost_effect": "increase" | "decrease" | "none",
    "stated_magnitude_min_pct": float,
    "stated_magnitude_max_pct": float
  }
}
Rules:
- If the note gives an explicit end date (e.g. 'from Feb 24 to Mar 8'), set effective_end_date to that exact date ('2025-03-08').
- If the note describes a single week (e.g. 'a festival week', 'about a week'), set effective_end_date to 6 days after effective_start_date.
- If the note is open-ended (e.g. 'starting this week'), set effective_end_date to null.
- If cost_effect is not 'increase', stated_magnitude_min_pct and stated_magnitude_max_pct MUST be 0.0.
"""


def _build_enriched_note(
    nid: str, ndate: datetime.date, applies: str, text: str, res: dict
) -> EnrichedContextNote:
    start_str = (
        res.get("effective_start_date")
        or res.get("start_date")
        or str(ndate)
    )
    end_str = res.get("effective_end_date") or res.get("end_date")
    effect = str(res.get("cost_effect", "none")).lower()
    if effect not in ("increase", "decrease", "none"):
        effect = "none"

    min_mag = float(res.get("stated_magnitude_min_pct", 0.0) or 0.0)
    max_mag = float(res.get("stated_magnitude_max_pct", 0.0) or 0.0)
    if effect != "increase":
        min_mag = 0.0
        max_mag = 0.0
    elif min_mag > max_mag:
        min_mag, max_mag = max_mag, min_mag

    start_date = datetime.strptime(start_str, "%Y-%m-%d").date()
    end_date = (
        datetime.strptime(end_str, "%Y-%m-%d").date()
        if end_str and str(end_str).lower() not in ("none", "null")
        else None
    )
    if end_date and end_date < start_date:
        end_date = start_date

    return EnrichedContextNote(
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
    had_api_failure = False

    # Build a single batched prompt for all 10 notes (saves 9 API calls and avoids RPM limits)
    notes_payload = []
    for _, row in df_notes.iterrows():
        nid = str(row["note_id"]).strip()
        ndate = str(row["date"]).strip()
        applies = str(row["applies_to"]).strip()
        text = str(row["note"]).strip()
        notes_payload.append(
            f"Note ID: {nid} | Published Date: {ndate} | Applies To: {applies} | Text: {text}"
        )

    batch_prompt = "Extract structured metadata for all notes below:\n\n" + "\n".join(notes_payload)
    batch_res: dict = {}
    if settings.LLM_PROVIDER != "stub":
        try:
            batch_res = llm_client.complete_json(
                BATCH_ENRICHMENT_SYSTEM_PROMPT, batch_prompt, context="enrich_all_notes_batch"
            )
        except (FreightWatcherError, ValueError, KeyError, TypeError) as e:
            logger.warning("Batch note enrichment failed (%s); falling back to per-note calls", e)

    for _, row in df_notes.iterrows():
        nid = str(row["note_id"]).strip()
        ndate = datetime.strptime(str(row["date"]).strip(), "%Y-%m-%d").date()
        applies = str(row["applies_to"]).strip()
        text = str(row["note"]).strip()

        try:
            if nid in batch_res and isinstance(batch_res[nid], dict):
                res = batch_res[nid]
            else:
                prompt = (
                    f"Note ID: {nid}\nPublished Date: {ndate}\n"
                    f"Applies To: {applies}\nText: {text}"
                )
                res = llm_client.complete_json(
                    BATCH_ENRICHMENT_SYSTEM_PROMPT, prompt, context=f"enrich_{nid}"
                )
                if nid in res and isinstance(res[nid], dict):
                    res = res[nid]
            enriched = _build_enriched_note(nid, ndate, applies, text, res)
        except (FreightWatcherError, ValueError, KeyError, TypeError) as e:
            had_api_failure = True
            logger.error(
                "Note enrichment failed for %s (%s); degrading safely to inert note",
                nid,
                e,
            )
            enriched = EnrichedContextNote(
                note_id=nid,
                note_date=ndate,
                applies_to=applies,
                note=text,
                effective_start_date=ndate,
                effective_end_date=ndate,
                cost_effect="none",
                stated_magnitude_min_pct=0.0,
                stated_magnitude_max_pct=0.0,
            )
        enriched_dict[nid] = enriched

    if not had_api_failure:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(
                {nid: note.model_dump(mode="json") for nid, note in enriched_dict.items()},
                f,
                indent=2,
            )
        logger.info("Enriched and cached %d context notes.", len(enriched_dict))
    else:
        logger.warning("Skipped writing note cache due to transient API failure.")

    return enriched_dict
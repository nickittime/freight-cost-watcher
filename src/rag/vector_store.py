from datetime import datetime

import chromadb

from src.config import get_settings
from src.logger import get_logger
from src.schemas import EnrichedContextNote

logger = get_logger(__name__)


class NoteVectorStore:
    """In-memory Chroma vector index with scope-complete, date-proximity ranked retrieval."""

    def __init__(self, notes: dict[str, EnrichedContextNote]) -> None:
        self.notes = notes
        self.client = chromadb.Client()
        existing_colls = [
            c.name if hasattr(c, "name") else str(c)
            for c in self.client.list_collections()
        ]
        if "context_notes" in existing_colls:
            self.client.delete_collection(name="context_notes")
        self.collection = self.client.create_collection(name="context_notes")
        self._populate()

    def _populate(self) -> None:
        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict] = []

        for nid, n in self.notes.items():
            ids.append(nid)
            doc = f"Note {nid} (Date: {n.note_date}, Route: {n.applies_to}): {n.note}"
            documents.append(doc)
            metadatas.append({
                "note_id": nid,
                "applies_to": n.applies_to,
                "date": str(n.note_date),
            })

        self.collection.add(ids=ids, documents=documents, metadatas=metadatas)
        logger.info("Populated in-memory vector store with %d context notes.", len(ids))

    def retrieve_candidates(
        self, route: str, week_of: str, top_k: int | None = None
    ) -> list[EnrichedContextNote]:
        """
        Guarantees inclusion of all scope-matching notes ('route' or 'All Routes'),
        ranked by date proximity to week_of, route specificity, and dense vector rank.
        """
        settings = get_settings()
        k = top_k or settings.RETRIEVAL_TOP_K

        try:
            target_date = datetime.strptime(week_of, "%Y-%m-%d").date()
        except ValueError:
            target_date = datetime(2025, 1, 1).date()

        query = f"Freight shipping cost rise disruptions on route {route} around week {week_of}"
        results = self.collection.query(
            query_texts=[query], n_results=len(self.notes)
        )
        dense_order = results["ids"][0] if (results and results["ids"]) else list(self.notes.keys())
        dense_rank = {nid: idx for idx, nid in enumerate(dense_order)}

        scope_notes = [
            n for n in self.notes.values() if n.applies_to in (route, "All Routes")
        ]
        scope_notes.sort(
            key=lambda n: (
                abs((target_date - n.note_date).days),
                0 if n.applies_to == route else 1,
                dense_rank.get(n.note_id, 999),
            )
        )

        selected = scope_notes[: max(k, len(scope_notes))]
        return selected
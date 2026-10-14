# src/rag/vector_store.py
import chromadb

from src.config import get_settings
from src.logger import get_logger
from src.schemas import EnrichedContextNote

logger = get_logger(__name__)


class NoteVectorStore:
    """In-memory Chroma vector index with recall-oriented candidate retrieval."""

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
        Recall-oriented candidate retrieval.

        Queries notes matching route scope ('route' or 'All Routes') ranked by relevance.
        """
        settings = get_settings()
        k = top_k or settings.RETRIEVAL_TOP_K

        query = f"Freight shipping cost rise disruptions on route {route} around week {week_of}"
        results = self.collection.query(
            query_texts=[query], n_results=min(k * 2, len(self.notes))
        )

        retrieved_ids: list[str] = []
        if results and results["ids"]:
            for nid in results["ids"][0]:
                note = self.notes[nid]
                if note.applies_to in (route, "All Routes"):
                    retrieved_ids.append(nid)
                if len(retrieved_ids) >= k:
                    break

        if len(retrieved_ids) < k and results and results["ids"]:
            for nid in results["ids"][0]:
                if nid not in retrieved_ids:
                    retrieved_ids.append(nid)
                if len(retrieved_ids) >= k:
                    break

        return [self.notes[nid] for nid in retrieved_ids]
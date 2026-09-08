"""
Extensie pentru chunking.py: generează embeddings cu sentence-transformers
(all-MiniLM-L6-v2) și le salvează persistent într-un ChromaDB local (SQLite pe disc).
"""

from typing import List, Dict, Any, Optional

import chromadb
from sentence_transformers import SentenceTransformer

from chunking import extrage_text_din_html, genereaza_chunkuri_finale

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_PERSIST_DIR = "./chroma_db"
DEFAULT_COLLECTION = "documente_legislative"


class EmbeddingStore:
    """Wrapper peste SentenceTransformer + ChromaDB (client persistent, stocare SQLite)."""

    def __init__(
        self,
        persist_dir: str = DEFAULT_PERSIST_DIR,
        collection_name: str = DEFAULT_COLLECTION,
        model_name: str = MODEL_NAME,
    ):
        self.model = SentenceTransformer(model_name)

        # PersistentClient scrie automat pe disc (chroma.sqlite3 + fișiere index)
        # in directorul indicat de persist_dir.
        self.client = chromadb.PersistentClient(path=persist_dir)

        # cosine e alegerea standard pentru embeddings de tip sentence-transformers
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def _embed(self, texte: List[str]):
        return self.model.encode(
            texte,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,
        ).tolist()

    def adauga_chunkuri(self, chunkuri: List[Dict[str, Any]], batch_size: int = 64):
        """Calculează embeddings pentru fiecare chunk și le inserează în colecție."""
        if not chunkuri:
            return

        for i in range(0, len(chunkuri), batch_size):
            batch = chunkuri[i : i + batch_size]

            ids = [f"{c['source']}::{c['chunk_index']}" for c in batch]
            texte = [c["text"] for c in batch]
            embeddings = self._embed(texte)

            # metadatele nu pot conține None în Chroma -> le curățăm
            metadatas = [
                {
                    "source": c.get("source") or "",
                    "chunk_index": c.get("chunk_index", 0),
                    "articol": c.get("articol") or "",
                }
                for c in batch
            ]

            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=texte,
                metadatas=metadatas,
            )

    def cauta(self, intrebare: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Căutare semantică prin similaritate cosine pe embeddings."""
        query_emb = self._embed([intrebare])
        rezultate = self.collection.query(
            query_embeddings=query_emb,
            n_results=top_k,
        )

        iesire = []
        for doc, meta, dist, _id in zip(
            rezultate["documents"][0],
            rezultate["metadatas"][0],
            rezultate["distances"][0],
            rezultate["ids"][0],
        ):
            iesire.append({
                "id": _id,
                "chunk_index": meta.get("chunk_index"),
                "text": doc,
                "source": meta.get("source"),
                "articol": meta.get("articol") or None,
                "score": 1 - dist,  # distanță cosine -> similaritate
            })
        return iesire


def indexeaza_fisier_html(
    nume_fisier: str,
    persist_dir: str = DEFAULT_PERSIST_DIR,
    collection_name: str = DEFAULT_COLLECTION,
    chunk_size: int = 220,
    overlap: int = 30,
) -> EmbeddingStore:
    """Pipeline complet: HTML -> text -> chunk-uri -> embeddings -> ChromaDB persistent."""
    with open(nume_fisier, "r", encoding="utf-8", errors="ignore") as f:
        continut_html = f.read()

    text_extras = extrage_text_din_html(continut_html)
    chunkuri = genereaza_chunkuri_finale(
        text_extras, sursa=nume_fisier, chunk_size=chunk_size, overlap=overlap
    )

    store = EmbeddingStore(persist_dir=persist_dir, collection_name=collection_name)
    # dacă fișierul a mai fost indexat anterior (colecția e persistentă pe disc),
    # nu recalculăm embeddings-urile inutil
    if store.collection.count() == 0:
        store.adauga_chunkuri(chunkuri)
    return store


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Utilizare: python embeddings_chroma.py <fisier.html> [intrebare]")
        sys.exit(1)

    fisier = sys.argv[1]
    store = indexeaza_fisier_html(fisier)
    print(f"Chunk-uri indexate în colecția '{store.collection.name}'.")

    if len(sys.argv) > 2:
        intrebare = sys.argv[2]
        for r in store.cauta(intrebare, top_k=5):
            print(f"[{r['score']:.3f}] {r['source']} | {r['articol']}\n{r['text'][:200]}...\n")
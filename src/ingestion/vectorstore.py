import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class VectorStore:
    """
    Vector store simplu, în memorie.
    Stochează chunk-uri + embeddings și permite căutare semantică.
    """

    def __init__(self):
        self.chunkuri = []
        self.embeddings = None

    def add(self, chunkuri_cu_embedding: list[dict]):
        """
        Adaugă chunk-urile și embeddings în store.
        """
        self.chunkuri = chunkuri_cu_embedding
        self.embeddings = np.array([c["embedding"] for c in chunkuri_cu_embedding])

    def search(self, query_embedding, top_k: int = 5) -> list[dict]:
        """
        Caută chunk-uri similare folosind similaritate cosinus.
        """
        if self.embeddings is None or len(self.chunkuri) == 0:
            return []

        query_embedding = np.array(query_embedding).reshape(1, -1)
        scoruri = cosine_similarity(query_embedding, self.embeddings)[0]

        top_idx = np.argsort(scoruri)[::-1][:top_k]

        rezultate = []
        for i in top_idx:
            rezultate.append({
                "id": self.chunkuri[i]["id"],
                "text": self.chunkuri[i]["text"],
                "lungime": self.chunkuri[i]["lungime"],
                "score": float(scoruri[i])
            })

        return rezultate

    def save_to_npz(self, path: str):
        """
        Salvează embeddings și metadate simple într-un fișier npz.
        Atenție: textul îl salvăm ca listă de obiecte.
        """
        if self.embeddings is None:
            raise ValueError("Nu există embeddings de salvat.")

        np.savez_compressed(
            path,
            embeddings=self.embeddings,
            chunkuri=np.array(self.chunkuri, dtype=object)
        )

    def load_from_npz(self, path: str):
        """
        Încarcă embeddings și chunk-uri dintr-un fișier npz.
        """
        data = np.load(path, allow_pickle=True)
        self.embeddings = data["embeddings"]
        self.chunkuri = list(data["chunkuri"])
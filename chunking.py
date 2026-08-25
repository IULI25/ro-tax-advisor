from bs4 import BeautifulSoup
import re
from collections import Counter
import math
from functools import lru_cache
from typing import List, Dict, Any

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


# ----------------------------
# CONFIGURARE MODEL
# ----------------------------
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _get_model():
    """
    Încarcă modelul de embeddings o singură dată.
    """
    return SentenceTransformer(EMBEDDING_MODEL)


def _embed_texts(texts: List[str]) -> np.ndarray:
    """
    Generează embeddings pentru o listă de texte.
    """
    model = _get_model()
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return np.array(embeddings, dtype="float32")


# ----------------------------
# 1. EXTRAGERE TEXT DIN HTML
# ----------------------------
def extrage_text_din_html(html: str) -> str:
    """
    Extrage textul curat dintr-un HTML.
    Elimină script, style și spații inutile.
    """
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]

    return "\n".join(lines)


# ----------------------------
# 2. TOKENIZARE (opțional, păstrată pentru compatibilitate)
# ----------------------------
def _tokenize(text: str):
    """
    Transformă textul în tokeni simpli.
    Păstrată pentru compatibilitate, dar nu mai este folosită în retrieval.
    """
    return re.findall(r"\w+", text.lower())


# ----------------------------
# 3. SPLIT ÎN CHUNK-URI
# ----------------------------
def _split_into_chunks(text: str, chunk_size: int = 800, overlap: int = 100):
    """
    Împarte textul în chunk-uri de lungime aproximativă.
    """
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)

        if end == len(words):
            break

        start = end - overlap

    return chunks


# ----------------------------
# 4. GENERARE CHUNK-URI FINALE
# ----------------------------
def genereaza_chunkuri_finale(text: str, sursa: str, chunk_size: int = 120, overlap: int = 20):
    """
    Generează chunk-uri cu metadate.
    """
    words = text.split()
    chunkuri = []
    start = 0
    idx = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_text = " ".join(words[start:end])

        chunkuri.append({
            "id": idx,
            "chunk_index": idx,
            "text": chunk_text,
            "source": sursa
        })

        idx += 1
        if end == len(words):
            break
        start = end - overlap

    return chunkuri


# ----------------------------
# 5. CONSTRUIRE INDEX FAISS
# ----------------------------
def _construieste_index_faiss(chunkuri: List[Dict[str, Any]]):
    """
    Construiește indexul FAISS pentru chunk-urile date.
    Returnează indexul și matricea de embeddings.
    """
    if not chunkuri:
        return None, None

    texts = [chunk["text"] for chunk in chunkuri]
    embeddings = _embed_texts(texts)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner Product; merge bine cu embeddings normalizate
    index.add(embeddings)

    return index, embeddings


# ----------------------------
# 6. SELECȚIE CHUNK-URI RELEVANTE
# ----------------------------
def selecteaza_chunkuri_relevante(chunkuri: list, intrebare: str, top_k: int = 5):
    """
    Selectează cele mai relevante chunk-uri pentru întrebare folosind embeddings + FAISS.
    """
    if not chunkuri:
        return []

    texts = [chunk["text"] for chunk in chunkuri]
    embeddings = _embed_texts(texts)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    query_embedding = _embed_texts([intrebare])

    k = min(top_k, len(chunkuri))
    scores, indices = index.search(query_embedding, k)

    rezultate = []
    for score, idx in zip(scores[0], indices[0]):
        chunk = chunkuri[idx].copy()
        chunk["score"] = float(score)
        rezultate.append(chunk)

    return rezultate
from bs4 import BeautifulSoup
import re
from functools import lru_cache
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


# ----------------------------
# CONFIGURARE MODEL
# ----------------------------
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Regex pentru detectarea începutului unui articol de lege
# (ex: "Articolul 1", "Art. 25", "ART. 3^1")
_ARTICOL_RE = re.compile(r"(?m)^\s*(Articolul|Art\.)\s+\d+", re.IGNORECASE)


@lru_cache(maxsize=1)
def _get_model():
    """Încarcă modelul de embeddings o singură dată (proces curent)."""
    return SentenceTransformer(EMBEDDING_MODEL)


def _embed_texts(texts: List[str]) -> np.ndarray:
    """Generează embeddings pentru o listă de texte."""
    model = _get_model()
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(embeddings, dtype="float32")


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
# 2. SPLIT PE ARTICOLE (cu fallback pe cuvinte)
# ----------------------------
def _split_pe_articole(text: str) -> List[str]:
    """
    Împarte textul la fiecare început de articol ("Articolul N" / "Art. N").
    Dacă nu găsește marcatori de articol, întoarce lista goală (fallback în apelant).
    """
    matches = list(_ARTICOL_RE.finditer(text))
    if len(matches) < 2:
        return []

    segmente = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        segment = text[start:end].strip()
        if segment:
            segmente.append(segment)
    return segmente


def _split_segment_in_bucati(segment: str, chunk_size: int, overlap: int) -> List[str]:
    """Împarte un segment (ex: un articol lung) în bucăți de `chunk_size` cuvinte, cu overlap."""
    words = segment.split()
    if not words:
        return []
    if len(words) <= chunk_size:
        return [segment]

    bucati = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        bucati.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = end - overlap
    return bucati


def _split_into_chunks(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    """
    Împarte textul în chunk-uri de lungime aproximativă (pe cuvinte).
    Păstrată pentru compatibilitate.
    """
    return _split_segment_in_bucati(text, chunk_size, overlap)


# ----------------------------
# 3. GENERARE CHUNK-URI FINALE
# ----------------------------
def genereaza_chunkuri_finale(
    text: str,
    sursa: str,
    chunk_size: int = 220,
    overlap: int = 30,
) -> List[Dict[str, Any]]:
    """
    Generează chunk-uri cu metadate. Pentru texte de lege, încearcă întâi
    să taie pe granițe de articol (rezultate mult mai relevante la retrieval
    decât o tăiere oarbă la N cuvinte), apoi sub-împarte articolele lungi.
    Dacă nu detectează articole, cade pe tăierea clasică pe cuvinte.
    """
    segmente = _split_pe_articole(text)
    if not segmente:
        segmente = [text]

    chunkuri = []
    idx = 0
    for segment in segmente:
        # titlul articolului, dacă există, folosit ca metadată suplimentară
        titlu_match = _ARTICOL_RE.match(segment)
        titlu = segment.splitlines()[0].strip() if titlu_match else None

        bucati = _split_segment_in_bucati(segment, chunk_size, overlap)
        for bucata in bucati:
            chunkuri.append({
                "id": idx,
                "chunk_index": idx,
                "text": bucata,
                "source": sursa,
                "articol": titlu,
            })
            idx += 1

    return chunkuri


# ----------------------------
# 4. INDEX FAISS (construit o singură dată, reutilizat)
# ----------------------------
def construieste_index(chunkuri: List[Dict[str, Any]]) -> Tuple[Optional[Any], Optional[np.ndarray]]:
    """
    Construiește indexul FAISS pentru chunk-urile date, o singură dată.
    Apelantul (main.py) trebuie să pună rezultatul în cache (st.cache_resource)
    și să-l refolosească la fiecare întrebare — NU reconstrui la fiecare query.
    """
    if not chunkuri:
        return None, None

    texts = [c["text"] for c in chunkuri]
    embeddings = _embed_texts(texts)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner Product; potrivit cu embeddings normalizate
    index.add(embeddings)

    return index, embeddings


# ----------------------------
# 5. SELECȚIE CHUNK-URI RELEVANTE
# ----------------------------
def selecteaza_chunkuri_relevante(
    chunkuri: List[Dict[str, Any]],
    intrebare: str,
    top_k: int = 5,
    index: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """
    Selectează cele mai relevante chunk-uri pentru întrebare folosind embeddings + FAISS.

    Dacă `index` este furnizat (construit anterior cu `construieste_index` și pus
    în cache), NU se mai recalculează embeddings pentru toate chunk-urile — se
    calculează doar embedding-ul întrebării. Asta transformă o căutare din
    "O(toate chunk-urile) la fiecare mesaj" în "O(1) la fiecare mesaj".
    """
    if not chunkuri:
        return []

    if index is None:
        # fallback: fără index precalculat, îl construim (lent, dar corect)
        index, _ = construieste_index(chunkuri)
        if index is None:
            return []

    query_embedding = _embed_texts([intrebare])

    k = min(top_k, len(chunkuri))
    scores, indices = index.search(query_embedding, k)

    rezultate = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        chunk = chunkuri[idx].copy()
        chunk["score"] = float(score)
        rezultate.append(chunk)

    return rezultate
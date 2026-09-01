"""
Extragere text din HTML, chunking pe articole de lege și căutare semantică
folosind embeddings Gemini (gemini-embedding-001), fără dependențe grele
precum sentence-transformers / PyTorch / FAISS.

IMPORTANT: `genai.configure(api_key=...)` trebuie apelat (în main.py) ÎNAINTE
de orice funcție din acest fișier care generează embeddings
(construieste_index / selecteaza_chunkuri_relevante), altfel API-ul Gemini
va refuza cererile.
"""

import re
import time
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
from bs4 import BeautifulSoup
import google.generativeai as genai


# ----------------------------
# CONFIGURARE MODEL EMBEDDING
# ----------------------------
EMBEDDING_MODEL = "models/gemini-embedding-001"
# 768 e suficient pentru un singur document de lege și ține indexul mic/rapid;
# modelul suportă și 1536 / 3072 dacă vrei mai multă precizie.
EMBEDDING_DIM = 768

# API-ul Gemini acceptă embed în batch, dar limităm dimensiunea cererii
# ca să evităm erori de tip "payload prea mare" / rate limit.
EMBEDDING_BATCH_SIZE = 90
# Câte reîncercări facem dacă lovim un rate limit (429) sau o eroare temporară.
EMBEDDING_MAX_RETRIES = 3
EMBEDDING_RETRY_DELAY_SEC = 5

# Regex pentru detectarea începutului unui articol de lege
# (ex: "Articolul 1", "Art. 25", "ART. 3^1")
_ARTICOL_RE = re.compile(r"(?m)^\s*(Articolul|Art\.)\s+\d+", re.IGNORECASE)


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
# 4. EMBEDDINGS (Gemini API)
# ----------------------------
def _embed_batch(texts: List[str], task_type: str) -> np.ndarray:
    """
    Apelează Gemini embed_content pentru un batch de texte (<= EMBEDDING_BATCH_SIZE),
    cu reîncercări simple în caz de eroare temporară / rate limit.
    """
    ultima_eroare: Optional[Exception] = None
    for incercare in range(EMBEDDING_MAX_RETRIES):
        try:
            rezultat = genai.embed_content(
                model=EMBEDDING_MODEL,
                content=texts,
                task_type=task_type,
                output_dimensionality=EMBEDDING_DIM,
            )
            embeddings = rezultat["embedding"]
            # Dacă am trimis un singur text, unele versiuni ale API-ului
            # întorc direct vectorul (listă de float-uri), nu o listă de vectori.
            if embeddings and isinstance(embeddings[0], (int, float)):
                embeddings = [embeddings]
            return np.asarray(embeddings, dtype="float32")
        except Exception as e:  # rate limit / eroare de rețea temporară
            ultima_eroare = e
            if incercare < EMBEDDING_MAX_RETRIES - 1:
                time.sleep(EMBEDDING_RETRY_DELAY_SEC)

    raise RuntimeError(
        f"Nu am putut genera embeddings după {EMBEDDING_MAX_RETRIES} încercări: {ultima_eroare}"
    ) from ultima_eroare


def _embed_texts(texts: List[str], task_type: str = "retrieval_document") -> np.ndarray:
    """
    Generează embeddings pentru o listă de texte folosind API-ul Gemini,
    trimițând cererile în batch-uri (EMBEDDING_BATCH_SIZE texte/cerere).

    task_type: "retrieval_document" pentru chunk-urile indexate,
               "retrieval_query" pentru întrebarea utilizatorului.
    """
    if not texts:
        return np.zeros((0, EMBEDDING_DIM), dtype="float32")

    toate = []
    for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[i:i + EMBEDDING_BATCH_SIZE]
        toate.append(_embed_batch(batch, task_type))
    embeddings = np.vstack(toate)

    # Normalizare L2 -> produsul scalar devine echivalent cu similaritatea cosinus.
    norme = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norme[norme == 0] = 1.0
    return embeddings / norme


# ----------------------------
# 5. INDEX ÎN MEMORIE (înlocuiește FAISS)
# ----------------------------
class IndexEmbeddings:
    """
    Index simplu, în memorie, peste o matrice de embeddings deja normalizate.
    Expune aceeași interfață `.search(query_embeddings, k) -> (scoruri, indici)`
    ca indexul FAISS folosit anterior, ca să nu fie nevoie de modificări în main.py.

    Pentru un singur document (legea 227/2015, câteva sute de chunk-uri),
    o căutare brute-force cu numpy e mai mult decât suficient de rapidă —
    nu are sens complexitatea suplimentară a FAISS.
    """

    def __init__(self, embeddings: np.ndarray):
        self.embeddings = embeddings  # (n_chunkuri, dim), deja normalizate

    def search(self, query_embeddings: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        if self.embeddings.shape[0] == 0:
            n_q = query_embeddings.shape[0]
            return np.zeros((n_q, 0), dtype="float32"), np.full((n_q, 0), -1, dtype="int64")

        # similaritate cosinus = produs scalar (vectorii sunt normalizați L2)
        scoruri_toate = query_embeddings @ self.embeddings.T  # (n_queries, n_chunkuri)
        k = min(k, scoruri_toate.shape[1])

        indici = np.argsort(-scoruri_toate, axis=1)[:, :k]
        scoruri = np.take_along_axis(scoruri_toate, indici, axis=1)
        return scoruri, indici


def construieste_index(chunkuri: List[Dict[str, Any]]) -> Tuple[Optional[IndexEmbeddings], Optional[np.ndarray]]:
    """
    Construiește indexul pentru chunk-urile date, o singură dată (calculează
    embeddings Gemini pentru toate chunk-urile). Apelantul (main.py) trebuie
    să pună rezultatul în cache (st.cache_resource) și să-l refolosească la
    fiecare întrebare — NU reconstrui la fiecare query (costă apeluri API).
    """
    if not chunkuri:
        return None, None

    texts = [c["text"] for c in chunkuri]
    embeddings = _embed_texts(texts, task_type="retrieval_document")
    index = IndexEmbeddings(embeddings)

    return index, embeddings


# ----------------------------
# 6. SELECȚIE CHUNK-URI RELEVANTE
# ----------------------------
def selecteaza_chunkuri_relevante(
    chunkuri: List[Dict[str, Any]],
    intrebare: str,
    top_k: int = 5,
    index: Optional[IndexEmbeddings] = None,
) -> List[Dict[str, Any]]:
    """
    Selectează cele mai relevante chunk-uri pentru întrebare folosind
    embeddings Gemini + căutare cosinus.

    Dacă `index` este furnizat (construit anterior cu `construieste_index` și pus
    în cache), NU se mai recalculează embeddings pentru toate chunk-urile — se
    calculează doar embedding-ul întrebării (1 apel API per întrebare, în loc
    de N apeluri la fiecare mesaj).
    """
    if not chunkuri:
        return []

    if index is None:
        # fallback: fără index precalculat, îl construim (costă N apeluri API)
        index, _ = construieste_index(chunkuri)
        if index is None:
            return []

    query_embedding = _embed_texts([intrebare], task_type="retrieval_query")

    k = min(top_k, len(chunkuri))
    scores, indices = index.search(query_embedding, k)

    rezultate = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        chunk = chunkuri[int(idx)].copy()
        chunk["score"] = float(score)
        rezultate.append(chunk)

    return rezultate
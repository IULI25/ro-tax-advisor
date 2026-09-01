import re
import time
import hashlib
import os
import pickle
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
from bs4 import BeautifulSoup
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted, GoogleAPIError


# ----------------------------
# CONFIGURARE MODEL EMBEDDING
# ----------------------------
# Numele oficial al modelului în SDK-ul Gemini (ex: "models/text-embedding-004" sau "models/gemini-embedding-001")
EMBEDDING_MODEL = "models/gemini-embedding-001"
EMBEDDING_DIM = 768

# Dimensiunea batch-ului redusă la 30 pentru a nu atinge limita de tokeni/cereri pe minut pe planul gratuit
EMBEDDING_BATCH_SIZE = 30
EMBEDDING_MAX_RETRIES = 5
EMBEDDING_INITIAL_RETRY_DELAY = 10.0

# Pauză proactivă de 2 secunde între batch-uri consecutive
BATCH_DELAY_SEC = 2.0

# Schimbă versiunea dacă modifici formatul chunkurilor/indexului.
INDEX_CACHE_VERSION = 1

# Regex pentru detectarea începutului unui articol de lege
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
    """Împarte un segment în bucăți de `chunk_size` cuvinte, cu overlap."""
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
    segmente = _split_pe_articole(text)
    if not segmente:
        segmente = [text]

    chunkuri = []
    idx = 0
    for segment in segmente:
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
    Apelează Gemini embed_content cu extragere dinamică a timpului de retry oferit de server.
    """
    delay = EMBEDDING_INITIAL_RETRY_DELAY
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
            if embeddings and isinstance(embeddings[0], (int, float)):
                embeddings = [embeddings]
            return np.asarray(embeddings, dtype="float32")

        except (ResourceExhausted, GoogleAPIError, Exception) as e:
            ultima_eroare = e
            if incercare < EMBEDDING_MAX_RETRIES - 1:
                # Căutăm secunde recomandate direct din mesajul de eroare al API-ului
                match = re.search(r"retry in (\d+(\.\d+)?)s", str(e), re.IGNORECASE)
                if match:
                    timp_asteptare = float(match.group(1)) + 1.0
                else:
                    timp_asteptare = delay
                    delay *= 2  # Exponential backoff

                print(
                    f"[Rate Limit] Așteptăm {timp_asteptare:.1f} secunde "
                    f"(Încercarea {incercare + 1}/{EMBEDDING_MAX_RETRIES})..."
                )
                time.sleep(timp_asteptare)
            else:
                break

    raise RuntimeError(
        f"Nu am putut genera embeddings după {EMBEDDING_MAX_RETRIES} încercări: {ultima_eroare}"
    ) from ultima_eroare


def _embed_texts(texts: List[str], task_type: str = "retrieval_document") -> np.ndarray:
    if not texts:
        return np.zeros((0, EMBEDDING_DIM), dtype="float32")

    toate = []
    total_batches = (len(texts) + EMBEDDING_BATCH_SIZE - 1) // EMBEDDING_BATCH_SIZE

    for idx, i in enumerate(range(0, len(texts), EMBEDDING_BATCH_SIZE)):
        batch = texts[i:i + EMBEDDING_BATCH_SIZE]
        toate.append(_embed_batch(batch, task_type))

        # Introducem o pauză controlată între batch-uri consecutive
        if idx < total_batches - 1:
            time.sleep(BATCH_DELAY_SEC)

    embeddings = np.vstack(toate)

    norme = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norme[norme == 0] = 1.0
    return embeddings / norme


# ----------------------------
# 5. INDEX ÎN MEMORIE
# ----------------------------
class IndexEmbeddings:
    def __init__(self, embeddings: np.ndarray):
        self.embeddings = embeddings

    def search(self, query_embeddings: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        if self.embeddings.shape[0] == 0:
            n_q = query_embeddings.shape[0]
            return np.zeros((n_q, 0), dtype="float32"), np.full((n_q, 0), -1, dtype="int64")

        scoruri_toate = query_embeddings @ self.embeddings.T
        k = min(k, scoruri_toate.shape[1])

        indici = np.argsort(-scoruri_toate, axis=1)[:, :k]
        scoruri = np.take_along_axis(scoruri_toate, indici, axis=1)
        return scoruri, indici


def construieste_index(chunkuri: List[Dict[str, Any]]) -> Tuple[Optional[IndexEmbeddings], Optional[np.ndarray]]:
    if not chunkuri:
        return None, None

    texts = [c["text"] for c in chunkuri]
    embeddings = _embed_texts(texts, task_type="retrieval_document")
    index = IndexEmbeddings(embeddings)

    return index, embeddings



# ----------------------------
# 6. CACHE PERSISTENT PE DISC
# ----------------------------
def hash_fisier(nume_fisier: str) -> str:
    """SHA-256 al documentului sursă; detectează orice modificare a HTML-ului."""
    sha = hashlib.sha256()
    with open(nume_fisier, "rb") as f:
        for bloc in iter(lambda: f.read(1024 * 1024), b""):
            sha.update(bloc)
    return sha.hexdigest()


def incarca_index_de_pe_disc(
    cale_cache: str,
    hash_sursa: str,
) -> Tuple[Optional[List[Dict[str, Any]]], Optional[IndexEmbeddings]]:
    """
    Încarcă indexul existent fără niciun apel Gemini.
    Returnează (None, None) dacă nu există sau nu mai este valid.
    """
    if not os.path.exists(cale_cache):
        return None, None

    try:
        with open(cale_cache, "rb") as f:
            date = pickle.load(f)

        if date.get("version") != INDEX_CACHE_VERSION:
            return None, None
        if date.get("source_hash") != hash_sursa:
            return None, None
        if date.get("embedding_model") != EMBEDDING_MODEL:
            return None, None
        if date.get("embedding_dim") != EMBEDDING_DIM:
            return None, None

        chunkuri = date["chunkuri"]
        embeddings = np.asarray(date["embeddings"], dtype="float32")

        if embeddings.ndim != 2 or len(chunkuri) != embeddings.shape[0]:
            return None, None

        return chunkuri, IndexEmbeddings(embeddings)

    except (OSError, EOFError, pickle.PickleError, KeyError, ValueError, TypeError):
        # Cache invalid/corupt -> îl reconstruim.
        return None, None


def salveaza_index_pe_disc(
    cale_cache: str,
    hash_sursa: str,
    chunkuri: List[Dict[str, Any]],
    index: IndexEmbeddings,
) -> None:
    """Salvează indexul atomic, pentru a evita un cache parțial dacă aplicația cade."""
    date = {
        "version": INDEX_CACHE_VERSION,
        "source_hash": hash_sursa,
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dim": EMBEDDING_DIM,
        "chunkuri": chunkuri,
        "embeddings": np.asarray(index.embeddings, dtype="float32"),
    }

    director = os.path.dirname(os.path.abspath(cale_cache))
    os.makedirs(director, exist_ok=True)

    cale_tmp = cale_cache + ".tmp"
    with open(cale_tmp, "wb") as f:
        pickle.dump(date, f, protocol=pickle.HIGHEST_PROTOCOL)

    os.replace(cale_tmp, cale_cache)


def incarca_sau_construieste_index(
    nume_fisier: str,
    cale_cache: str = "index_cache.pkl",
) -> Tuple[List[Dict[str, Any]], Optional[IndexEmbeddings], bool]:
    """
    Prima rulare:
      HTML -> chunkuri -> Gemini embeddings -> index_cache.pkl

    Următoarele rulări:
      index_cache.pkl -> memorie

    Altfel spus, Gemini NU este apelat pentru embeddings la restart dacă
    documentul și configurația embeddingului sunt neschimbate.

    Returnează:
      (chunkuri, index, folosit_cache)
    """
    hash_sursa = hash_fisier(nume_fisier)

    chunkuri_cache, index_cache = incarca_index_de_pe_disc(
        cale_cache,
        hash_sursa,
    )

    if chunkuri_cache is not None and index_cache is not None:
        return chunkuri_cache, index_cache, True

    with open(nume_fisier, "r", encoding="utf-8", errors="ignore") as f:
        continut_html = f.read()

    text_extras = extrage_text_din_html(continut_html)
    chunkuri = genereaza_chunkuri_finale(
        text_extras,
        sursa=nume_fisier,
    )

    index, _ = construieste_index(chunkuri)

    if index is None:
        raise RuntimeError("Nu s-a putut construi indexul pentru document.")

    salveaza_index_pe_disc(
        cale_cache,
        hash_sursa,
        chunkuri,
        index,
    )

    return chunkuri, index, False


# ----------------------------
# 6. SELECȚIE CHUNK-URI RELEVANTE
# ----------------------------
def selecteaza_chunkuri_relevante(
    chunkuri: List[Dict[str, Any]],
    intrebare: str,
    top_k: int = 5,
    index: Optional[IndexEmbeddings] = None,
) -> List[Dict[str, Any]]:
    if not chunkuri:
        return []

    if index is None:
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
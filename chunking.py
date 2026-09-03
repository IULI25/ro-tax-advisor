import re
import math
from collections import Counter
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from bs4 import BeautifulSoup


# ----------------------------
# 1. EXTRAGERE TEXT DIN HTML
# ----------------------------
def extrage_text_din_html(html: str) -> str:
    """Extrage textul curat dintr-un HTML, eliminând tag-urile irelevante."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


# ----------------------------
# 2. SPLIT PE ARTICOLE
# ----------------------------
_ARTICOL_RE = re.compile(r"(?m)^\s*(Articolul|Art\.)\s+\d+", re.IGNORECASE)

def _split_pe_articole(text: str) -> List[str]:
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
# 4. MOTOR VECTORIAL LOCAL BM25 / TF-IDF
# ----------------------------
def _tokenize(text: str) -> List[str]:
    """Tokenizare rapidă în litere mici fără semne de punctuație."""
    return re.findall(r"\w+", text.lower())


class VectorizerLocalBM25:
    """Motor de căutare vectorial local bazat pe BM25 ultra-rapid (fără dependențe de rețea)."""
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_len = []
        self.avgdl = 0.0
        self.doc_freqs = []
        self.idf = {}
        self.corpus_size = 0

    def fit(self, documents: List[str]):
        self.corpus_size = len(documents)
        if self.corpus_size == 0:
            return

        df = Counter()
        self.doc_len = []
        self.doc_freqs = []

        for doc in documents:
            tokens = _tokenize(doc)
            self.doc_len.append(len(tokens))
            freqs = Counter(tokens)
            self.doc_freqs.append(freqs)
            df.update(freqs.keys())

        self.avgdl = sum(self.doc_len) / self.corpus_size if self.corpus_size > 0 else 1.0

        for word, freq in df.items():
            # BM25 IDF Formula
            self.idf[word] = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        query_tokens = _tokenize(query)
        if not query_tokens or self.corpus_size == 0:
            return []

        scores = np.zeros(self.corpus_size, dtype="float32")

        for token in query_tokens:
            if token not in self.idf:
                continue
            idf_val = self.idf[token]
            for idx, freqs in enumerate(self.doc_freqs):
                freq = freqs.get(token, 0)
                if freq > 0:
                    numerator = freq * (self.k1 + 1)
                    denominator = freq + self.k1 * (1 - self.b + self.b * (self.doc_len[idx] / self.avgdl))
                    scores[idx] += idf_val * (numerator / denominator)

        indices = np.argsort(-scores)[:top_k]
        return [(int(i), float(scores[i])) for i in indices if scores[i] > 0]


# ----------------------------
# 5. ÎNCĂRCARE ȘI INDEXARE INSTANTĂ
# ----------------------------
def incarca_si_indexeaza_html(nume_fisier: str) -> Tuple[List[Dict[str, Any]], VectorizerLocalBM25]:
    """Citesște direct HTML-ul și construiește vectorii local în milisecunde."""
    with open(nume_fisier, "r", encoding="utf-8", errors="ignore") as f:
        continut_html = f.read()

    text_extras = extrage_text_din_html(continut_html)
    chunkuri = genereaza_chunkuri_finale(text_extras, sursa=nume_fisier)

    vectorizer = VectorizerLocalBM25()
    vectorizer.fit([c["text"] for c in chunkuri])

    return chunkuri, vectorizer


def selecteaza_chunkuri_relevante(
    chunkuri: List[Dict[str, Any]],
    intrebare: str,
    top_k: int = 5,
    vectorizer: Optional[VectorizerLocalBM25] = None,
) -> List[Dict[str, Any]]:
    if not chunkuri or vectorizer is None:
        return []

    rezultate = []
    top_matches = vectorizer.search(intrebare, top_k=top_k)

    for idx, scor in top_matches:
        chunk = chunkuri[idx].copy()
        chunk["score"] = scor
        rezultate.append(chunk)

    return rezultate
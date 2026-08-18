from bs4 import BeautifulSoup
import re
from collections import Counter
import math


def extrage_text_din_html(html: str) -> str:
    """
    Extrage textul curat dintr-un HTML.
    Elimină script, style și spații inutile.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Elimină elementele care nu sunt utile pentru conținut
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]

    return "\n".join(lines)


def _tokenize(text: str):
    """
    Transformă textul în tokeni simpli.
    """
    return re.findall(r"\w+", text.lower())


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


def genereaza_chunkuri_finale(text: str, chunk_size: int = 800, overlap: int = 100):
    """
    Creează lista finală de chunk-uri sub formă de dicționare.
    Fiecare chunk conține textul și un id.
    """
    raw_chunks = _split_into_chunks(text, chunk_size=chunk_size, overlap=overlap)

    chunkuri = []
    for i, chunk in enumerate(raw_chunks):
        chunkuri.append({
            "id": i,
            "text": chunk
        })

    return chunkuri


def _cosine_similarity(text1: str, text2: str) -> float:
    """
    Calculează similaritatea cosine între două texte, pe baza frecvenței cuvintelor.
    """
    tokens1 = _tokenize(text1)
    tokens2 = _tokenize(text2)

    if not tokens1 or not tokens2:
        return 0.0

    c1 = Counter(tokens1)
    c2 = Counter(tokens2)

    all_terms = set(c1) | set(c2)

    dot = sum(c1[t] * c2[t] for t in all_terms)
    norm1 = math.sqrt(sum(v * v for v in c1.values()))
    norm2 = math.sqrt(sum(v * v for v in c2.values()))

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot / (norm1 * norm2)


def selecteaza_chunkuri_relevante(chunkuri: list, intrebare: str, top_k: int = 5):
    """
    Selectează cele mai relevante chunk-uri pentru întrebare.
    Se bazează pe similaritate lexicală simplă.
    """
    if not chunkuri:
        return []

    scoruri = []
    for chunk in chunkuri:
        scor = _cosine_similarity(chunk["text"], intrebare)
        scoruri.append((scor, chunk))

    scoruri.sort(key=lambda x: x[0], reverse=True)

    return [chunk for scor, chunk in scoruri[:top_k]]
import re
from typing import List, Dict


def chunk_dupa_articole(text: str) -> List[str]:
    """
    Împarte textul pe baza unor marcatori de tip:
    - Articolul 1
    - ART. 1
    - ART 1

    Dacă nu găsește structură, întoarce lista cu textul original.
    """
    pattern = re.compile(
        r"(?=^\s*(Articolul\s+\d+|ART\.?\s*\d+))",
        re.MULTILINE | re.IGNORECASE
    )

    parti = pattern.split(text)

    chunkuri = []
    buffer = ""

    for parte in parti:
        if not parte or not parte.strip():
            continue

        if re.match(r"^\s*(Articolul\s+\d+|ART\.?\s*\d+)", parte, re.IGNORECASE):
            if buffer.strip():
                chunkuri.append(buffer.strip())
            buffer = parte
        else:
            buffer += parte

    if buffer.strip():
        chunkuri.append(buffer.strip())

    return chunkuri if chunkuri else [text]


def chunk_generic(text: str, marime_max: int = 1000, overlap: int = 150) -> List[str]:
    """
    Chunking generic pe caractere, cu overlap.
    Folosit ca fallback pentru texte fără structură clară sau pentru chunk-uri prea lungi.
    """
    if marime_max <= 0:
        raise ValueError("marime_max trebuie să fie > 0.")
    if overlap < 0:
        raise ValueError("overlap trebuie să fie >= 0.")
    if overlap >= marime_max:
        raise ValueError("overlap trebuie să fie mai mic decât marime_max.")

    chunkuri = []
    start = 0
    lungime = len(text)

    while start < lungime:
        end = min(start + marime_max, lungime)
        chunkuri.append(text[start:end])

        if end == lungime:
            break

        start = end - overlap

    return chunkuri


def genereaza_chunkuri_finale(text: str, marime_max: int = 1000, overlap: int = 150) -> List[Dict]:
    """
    Strategie combinată:
    - încearcă împărțirea pe articole
    - dacă un chunk este prea lung, îl sparge generic
    - întoarce o listă de dicționare cu id, text, lungime
    """
    chunkuri_articole = chunk_dupa_articole(text)

    # Dacă nu există structură reală și textul este mare, folosește direct chunking generic
    if len(chunkuri_articole) == 1 and len(text) > marime_max:
        chunkuri_articole = chunk_generic(text, marime_max=marime_max, overlap=overlap)

    rezultat = []

    for i, chunk in enumerate(chunkuri_articole):
        if len(chunk) <= marime_max:
            rezultat.append({
                "id": f"chunk_{i}",
                "text": chunk,
                "lungime": len(chunk)
            })
        else:
            subchunkuri = chunk_generic(chunk, marime_max=marime_max, overlap=overlap)
            for j, sub in enumerate(subchunkuri):
                rezultat.append({
                    "id": f"chunk_{i}_{j}",
                    "text": sub,
                    "lungime": len(sub)
                })

    return rezultat


def selecteaza_chunkuri_relevante(cuvinte_cheie: List[str], chunkuri: List[Dict], top_k: int = 5) -> List[Dict]:
    """
    Variantă lexicală simplă de relevanță, bazată pe apariția cuvintelor-cheie.
    Nu este semantică, dar poate fi utilă ca fallback.

    Scorul este numărul de apariții ale cuvintelor-cheie în chunk.
    """
    rezultate = []

    cuvinte_cheie_lower = [c.lower() for c in cuvinte_cheie]

    for chunk in chunkuri:
        text_lower = chunk["text"].lower()
        scor = sum(text_lower.count(cuvant) for cuvant in cuvinte_cheie_lower)

        if scor > 0:
            rezultate.append({
                "id": chunk["id"],
                "text": chunk["text"],
                "lungime": chunk["lungime"],
                "score": scor
            })

    rezultate.sort(key=lambda x: x["score"], reverse=True)
    return rezultate[:top_k]
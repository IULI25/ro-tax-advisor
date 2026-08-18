from bs4 import BeautifulSoup
import re
 
 
def extrage_text_din_html(html: str) -> str:
    """Extrage text curat dintr-un string HTML (fie citit din fișier, fie descărcat de pe web)."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    linii = [l.strip() for l in text.splitlines()]
    linii = [l for l in linii if l]
    return "\n".join(linii)
 
 
def extrage_text_din_fisier(cale_fisier: str, encoding: str = "utf-8") -> str:
    """Extrage text dintr-un fișier HTML local."""
    with open(cale_fisier, "r", encoding=encoding, errors="ignore") as f:
        html = f.read()
    return extrage_text_din_html(html)
 
 
def chunk_dupa_articole(text: str) -> list[str]:
    """Împarte textul pe baza structurii 'Articolul X' / 'ART. X' (util pentru legi)."""
    pattern = re.compile(r'(?=^\s*(Articolul\s+\d+|ART\.\s*\d+))', re.MULTILINE | re.IGNORECASE)
    parti = pattern.split(text)
 
    chunkuri = []
    buffer = ""
    for parte in parti:
        if not parte or not parte.strip():
            continue
        if re.match(r'^\s*(Articolul\s+\d+|ART\.\s*\d+)', parte, re.IGNORECASE):
            if buffer.strip():
                chunkuri.append(buffer.strip())
            buffer = parte
        else:
            buffer += parte
    if buffer.strip():
        chunkuri.append(buffer.strip())
 
    return chunkuri if chunkuri else [text]
 
 
def chunk_generic(text: str, marime_max: int = 1000, overlap: int = 150) -> list[str]:
    """Chunking generic pe caractere cu overlap, pentru texte fără structură de articole."""
    chunkuri = []
    start = 0
    lungime = len(text)
    while start < lungime:
        end = start + marime_max
        chunkuri.append(text[start:end])
        start = end - overlap
    return chunkuri
 
 
def genereaza_chunkuri_finale(text: str, marime_max: int = 1000, overlap: int = 150) -> list[dict]:
    """
    Strategie combinată: încearcă întâi împărțirea pe articole; dacă un
    'articol' rezultat e prea lung, îl mai sparge o dată cu overlap.
    Dacă textul nu are deloc structură de articole, cade automat pe chunking generic.
    """
    chunkuri_articole = chunk_dupa_articole(text)
 
    # dacă nu s-a găsit nicio structură de articole (un singur chunk == tot textul),
    # și textul e lung, trecem direct pe chunking generic
    if len(chunkuri_articole) == 1 and len(text) > marime_max:
        chunkuri_articole = chunk_generic(text, marime_max, overlap)
 
    rezultat = []
    for i, chunk in enumerate(chunkuri_articole):
        if len(chunk) <= marime_max:
            rezultat.append({"id": f"chunk_{i}", "text": chunk, "lungime": len(chunk)})
        else:
            for j, sub in enumerate(chunk_generic(chunk, marime_max, overlap)):
                rezultat.append({"id": f"chunk_{i}_{j}", "text": sub, "lungime": len(sub)})
 
    return rezultat
 
 
def selecteaza_chunkuri_relevante(chunkuri: list[dict], intrebare: str, top_k: int = 5) -> list[dict]:
    """
    Retrieval simplu bazat pe suprapunere de cuvinte (fără embeddings/vector DB).
    Suficient pentru documente de dimensiune moderată; pentru corpus mare,
    înlocuiește cu embeddings (ex. sentence-transformers) + similaritate cosinus.
    """
    cuvinte_intrebare = set(re.findall(r'\w+', intrebare.lower()))
    if not cuvinte_intrebare:
        return chunkuri[:top_k]
 
    scoruri = []
    for c in chunkuri:
        cuvinte_chunk = set(re.findall(r'\w+', c["text"].lower()))
        scor = len(cuvinte_intrebare & cuvinte_chunk)
        scoruri.append((scor, c))
 
    scoruri.sort(key=lambda x: x[0], reverse=True)
    return [c for scor, c in scoruri[:top_k] if scor > 0] or chunkuri[:top_k]
from sentence_transformers import SentenceTransformer


def incarca_model(nume_model: str = "all-MiniLM-L6-v2"):
    """
    Încarcă modelul de embeddings.
    Puteți schimba modelul cu unul mai performant, dacă doriți.
    """
    model = SentenceTransformer(nume_model)
    return model


def genereaza_embeddinguri(chunkuri: list[dict], model) -> list[dict]:
    """
    Generează embedding pentru fiecare chunk și returnează lista îmbogățită.
    """
    if not chunkuri:
        return []

    texte = [chunk["text"] for chunk in chunkuri]
    vectori = model.encode(texte, show_progress_bar=True)

    rezultat = []
    for chunk, vector in zip(chunkuri, vectori):
        rezultat.append({
            "id": chunk["id"],
            "text": chunk["text"],
            "lungime": chunk["lungime"],
            "embedding": vector
        })

    return rezultat


def genereaza_embedding_pentru_text(text: str, model):
    """
    Generează embedding pentru un singur text.
    """
    return model.encode([text])[0]
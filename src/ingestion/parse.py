from bs4 import BeautifulSoup


def extrage_text_din_html(html: str) -> str:
    """
    Extrage text curat dintr-un string HTML.
    Elimină script-urile și style-urile, apoi normalizează liniile goale.
    """
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    linii = [linia.strip() for linia in text.splitlines()]
    linii = [linia for linia in linii if linia]

    return "\n".join(linii)


def extrage_text_din_fisier(cale_fisier: str, encoding: str = "utf-8") -> str:
    """
    Citește un fișier HTML local și extrage textul curat din el.
    """
    with open(cale_fisier, "r", encoding=encoding, errors="ignore") as f:
        html = f.read()

    return extrage_text_din_html(html)
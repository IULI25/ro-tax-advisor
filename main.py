import streamlit as st
import google.generativeai as genai

from chunking import (
    extrage_text_din_html,
    genereaza_chunkuri_finale,
    selecteaza_chunkuri_relevante
)

GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

st.set_page_config(page_title="Agent AI din pagină web", page_icon="🤖", layout="centered")
st.title("🤖 Consilier AI")

api_key = GEMINI_API_KEY
fisier_html_local = "Legea_nr.227_2015.html"

with st.sidebar:
    st.header("Setări")
    if api_key:
        st.success("✅ Cheie API încărcată.")
    else:
        st.error(
            "❌ Nu am găsit GEMINI_API_KEY.\n\n"
            "Adaugă-l în `.streamlit/secrets.toml` sau în configurația Streamlit."
        )
    model_name = st.selectbox("Model", ["gemini-3.5-flash", "gemini-3.5-flash-lite"], index=0)

# ---------- State ----------
if "chunkuri" not in st.session_state:
    st.session_state.chunkuri = []
if "istoric" not in st.session_state:
    st.session_state.istoric = []
if "pagina_incarcata" not in st.session_state:
    st.session_state.pagina_incarcata = False

# ---------- Încărcare fișier local ----------
@st.cache_data
def proceseaza_fisier_local(nume_fisier: str):
    with open(nume_fisier, "r", encoding="utf-8", errors="ignore") as f:
        continut_html = f.read()

    text_extras = extrage_text_din_html(continut_html)

    # Recomandat: genereaza_chunkuri_finale să returneze dicționare cu text + source + chunk_index
    chunkuri = genereaza_chunkuri_finale(text_extras, sursa=nume_fisier)

    return chunkuri, text_extras

# Încarcă automat fișierul local la pornirea aplicației
if not st.session_state.pagina_incarcata:
    try:
        st.session_state.chunkuri, text_extras = proceseaza_fisier_local(fisier_html_local)
        st.session_state.pagina_incarcata = True
        st.success(f"✅ Fișierul `{fisier_html_local}` a fost încărcat cu succes.")
        st.write(f"Au fost generate **{len(st.session_state.chunkuri)} chunk-uri**.")
    except FileNotFoundError:
        st.error(f"❌ Nu am găsit fișierul `{fisier_html_local}` în directorul proiectului.")
    except Exception as e:
        st.error(f"❌ Eroare la procesarea fișierului HTML: {e}")

# ---------- Funcția de răspuns ----------
def raspunde(api_key: str, model_name: str, chunkuri: list, intrebare: str, istoric: list) -> tuple:
    """
    Returnează: (răspuns, chunkuri_relevante)
    """
    chunkuri_relevante = selecteaza_chunkuri_relevante(chunkuri, intrebare, top_k=5)

    context_piese = []
    for c in chunkuri_relevante:
        sursa = c.get("source", "necunoscut")
        idx = c.get("chunk_index", c.get("id", -1))
        scor = c.get("score", 0.0)
        text = c.get("text", "")

        context_piese.append(
            f"[Sursă: {sursa} | Chunk: {idx} | Scor: {scor:.4f}]\n{text}"
        )

    context = "\n\n---\n\n".join(context_piese)

    mesaje_istoric = ""
    for q, a in istoric[-3:]:
        mesaje_istoric += f"Întrebare anterioară: {q}\nRăspuns anterior: {a}\n\n"

    prompt = f"""Ai la dispoziție următoarele fragmente extrase din fișiere locale:

---
{context}
---

{mesaje_istoric}
Instrucțiuni:
- Răspunde exclusiv pe baza conținutului de mai sus.
- Dacă informația nu există în fragmentele oferite, spune clar că nu ai găsit răspunsul în fișier.
- Citează, dacă este posibil, sursa fragmentului folosit.

Întrebare: {intrebare}
"""

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    response = model.generate_content(prompt)

    return response.text, chunkuri_relevante

# ---------- UI: întrebări ----------
st.subheader("💬 Întreabă agentul")

intrebare = st.text_input("Întrebarea ta", placeholder="Ex: Care sunt orele de program?")
intreaba_btn = st.button("Trimite întrebarea")

if intreaba_btn:
    if not api_key:
        st.warning("Cheia API lipsește.")
    elif not st.session_state.chunkuri:
        st.warning("Fișierul HTML nu a fost încărcat corect.")
    elif not intrebare:
        st.warning("Scrie o întrebare.")
    else:
        with st.spinner("Agentul gândește..."):
            try:
                raspuns, chunkuri_relevante = raspunde(
                    api_key,
                    model_name,
                    st.session_state.chunkuri,
                    intrebare,
                    st.session_state.istoric
                )

                st.session_state.istoric.append((intrebare, raspuns))

                st.markdown("### Răspuns")
                st.write(raspuns)

                st.markdown("### Surse folosite")
                for c in chunkuri_relevante:
                    st.write(
                        f"- **Sursă:** {c.get('source', 'necunoscut')} | "
                        f"**Chunk:** {c.get('chunk_index', c.get('id', -1))} | "
                        f"**Scor:** {c.get('score', 0.0):.4f}"
                    )
                    st.caption(c.get("text", "")[:500] + ("..." if len(c.get("text", "")) > 500 else ""))

            except Exception as e:
                st.error(f"Eroare: {e}")

# ---------- Istoric conversație ----------
if st.session_state.istoric:
    st.divider()
    st.subheader("🗂️ Istoric conversație")
    for q, a in reversed(st.session_state.istoric):
        st.markdown(f"**Tu:** {q}")
        st.markdown(f"**Agent:** {a}")
        st.markdown("---")
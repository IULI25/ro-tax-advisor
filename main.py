import streamlit as st
import google.generativeai as genai

from chunking import (
    incarca_si_indexeaza_html,
    selecteaza_chunkuri_relevante,
)

st.set_page_config(page_title="Agent AI din pagină web", page_icon="🤖", layout="centered")
st.title("🤖 Consilier AI")

FISIER_HTML_LOCAL = "Legea_nr.227_2015.html"
MODELE_DISPONIBILE = ["gemini-1.5-flash", "gemini-1.5-pro"]

api_key = st.secrets.get("GEMINI_API_KEY")

with st.sidebar:
    st.header("Setări")
    if api_key:
        st.success("✅ Cheie API încărcată.")
    else:
        st.error(
            "❌ Nu am găsit GEMINI_API_KEY.\n\n"
            "Adaugă-l în `.streamlit/secrets.toml` sau în configurația Streamlit."
        )
    model_name = st.selectbox("Model", MODELE_DISPONIBILE, index=0)
    top_k = st.slider("Câte fragmente să folosească", 3, 10, 5)
    if st.button("🗑️ Golește istoricul conversației"):
        st.session_state.istoric = []
        st.rerun()

if not api_key:
    st.error(
        "❌ Nu am găsit GEMINI_API_KEY.\n\n"
        "Adaugă-l în `.streamlit/secrets.toml` sau în configurația Streamlit."
    )
    st.stop()

genai.configure(api_key=api_key)

# ---------- Cache rapid în RAM cu Streamlit ----------
@st.cache_resource(show_spinner=False)
def obtine_index_local(nume_fisier: str):
    return incarca_si_indexeaza_html(nume_fisier)

# ---------- State ----------
if "istoric" not in st.session_state:
    st.session_state.istoric = []  # listă de (intrebare, raspuns, surse)

# ---------- Încărcare inițială instantanee ----------
try:
    with st.spinner("Se încarcă documentul și se generează vectorii locali..."):
        chunkuri, vectorizer = obtine_index_local(FISIER_HTML_LOCAL)

    st.success("⚡ Document indexat local instant (0 API Latency)!")
    st.caption(f"📄 `{FISIER_HTML_LOCAL}` — {len(chunkuri)} fragmente indexate.")

except FileNotFoundError:
    st.error(f"❌ Nu am găsit fișierul `{FISIER_HTML_LOCAL}` în directorul proiectului.")
    st.stop()
except Exception as e:
    st.error(f"❌ Eroare la procesarea fișierului HTML: {e}")
    st.stop()


# ---------- Funcția de răspuns ----------
def raspunde(model_name: str, chunkuri: list, vectorizer, intrebare: str, istoric: list, top_k: int) -> tuple:
    """Returnează: (răspuns, chunkuri_relevante)."""
    chunkuri_relevante = selecteaza_chunkuri_relevante(chunkuri, intrebare, top_k=top_k, vectorizer=vectorizer)

    context_piese = []
    for c in chunkuri_relevante:
        sursa = c.get("source", "necunoscut")
        idx = c.get("chunk_index", c.get("id", -1))
        scor = c.get("score", 0.0)
        articol = c.get("articol")
        text = c.get("text", "")
        etichetă = f"[Sursă: {sursa} | {articol or f'Chunk {idx}'} | Scor: {scor:.4f}]"
        context_piese.append(f"{etichetă}\n{text}")

    context = "\n\n---\n\n".join(context_piese)

    mesaje_istoric = ""
    for q, a, _ in istoric[-3:]:
        mesaje_istoric += f"Întrebare anterioară: {q}\nRăspuns anterior: {a}\n\n"

    prompt = f"""Ai la dispoziție următoarele fragmente extrase din fișiere locale:

---
{context}
---

{mesaje_istoric}
Instrucțiuni:
- Răspunde exclusiv pe baza conținutului de mai sus.
- Dacă informația nu există în fragmentele oferite, spune clar că nu ai găsit răspunsul în fișier.
- Citează, dacă este posibil, articolul sau sursa fragmentului folosit.

Întrebare: {intrebare}
"""

    model = genai.GenerativeModel(model_name)
    response = model.generate_content(prompt)

    return response.text, chunkuri_relevante


# ---------- UI: chat ----------
st.subheader("💬 Întreabă agentul")

for q, a, surse in st.session_state.istoric:
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        st.markdown(a)
        if surse:
            with st.expander("Surse folosite"):
                for c in surse:
                    st.write(
                        f"- **{c.get('articol') or 'Chunk ' + str(c.get('chunk_index', c.get('id', -1)))}** "
                        f"| Scor: {c.get('score', 0.0):.4f}"
                    )
                    st.caption(c.get("text", "")[:500] + ("..." if len(c.get("text", "")) > 500 else ""))

intrebare = st.chat_input("Ex: Care este cota standard de TVA?")

if intrebare:
    if not api_key:
        st.warning("Cheia API lipsește.")
    else:
        with st.chat_message("user"):
            st.markdown(intrebare)

        with st.chat_message("assistant"):
            with st.spinner("Agentul gândește..."):
                try:
                    raspuns, chunkuri_relevante = raspunde(
                        model_name, chunkuri, vectorizer, intrebare, st.session_state.istoric, top_k
                    )
                    st.markdown(raspuns)
                    if chunkuri_relevante:
                        with st.expander("Surse folosite"):
                            for c in chunkuri_relevante:
                                st.write(
                                    f"- **{c.get('articol') or 'Chunk ' + str(c.get('chunk_index', c.get('id', -1)))}** "
                                    f"| Scor: {c.get('score', 0.0):.4f}"
                                )
                                st.caption(c.get("text", "")[:500] + ("..." if len(c.get("text", "")) > 500 else ""))

                    st.session_state.istoric.append((intrebare, raspuns, chunkuri_relevante))
                except Exception as e:
                    st.error(f"Eroare: {e}")
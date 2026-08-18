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
fisier_html_local = "pagina.html"

with st.sidebar:
    st.header("Setări")
    if api_key:
        st.success("✅ Cheie API încărcată.")
    else:
        st.error(
            "❌ Nu am găsit GEMINI_API_KEY.\n\n"
            "Adaugă-l în `.streamlit/secrets.toml` sau în configurația Streamlit."
        )
    model = st.selectbox("Model", ["gemini-2.0-flash", "gemini-2.0-flash-lite"], index=0)

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
    chunkuri = genereaza_chunkuri_finale(text_extras)
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
def raspunde(api_key: str, model_name: str, chunkuri: list, intrebare: str, istoric: list) -> str:
    chunkuri_relevante = selecteaza_chunkuri_relevante(chunkuri, intrebare, top_k=5)
    context = "\n\n---\n\n".join(c["text"] for c in chunkuri_relevante)

    mesaje_istoric = ""
    for q, a in istoric[-3:]:
        mesaje_istoric += f"Întrebare anterioară: {q}\nRăspuns anterior: {a}\n\n"

    prompt = f"""Ai la dispoziție următoarele fragmente extrase din fișierul HTML local:

---
{context}
---

{mesaje_istoric}
Pe baza EXCLUSIV a conținutului de mai sus, răspunde la întrebarea de mai jos.
Dacă informația nu se găsește în conținut, spune clar că nu ai găsit răspunsul în fișier.

Întrebare: {intrebare}
"""

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    response = model.generate_content(prompt)
    return response.text

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
                raspuns = raspunde(
                    api_key,
                    model,
                    st.session_state.chunkuri,
                    intrebare,
                    st.session_state.istoric
                )
                st.session_state.istoric.append((intrebare, raspuns))
                st.markdown("### Răspuns")
                st.write(raspuns)
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
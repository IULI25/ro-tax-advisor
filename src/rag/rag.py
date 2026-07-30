Y
import streamlit as st
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
from config import GEMINI_API_KEY
 
st.set_page_config(page_title="Agent AI din pagină web", page_icon="🤖", layout="centered")
 
st.title("🤖 Agent AI - răspunde din conținutul unei pagini web")
st.write(
    "Introdu un link, apoi pune întrebări. Agentul va răspunde "
    "folosind exclusiv conținutul extras de pe acea pagină."
)
 
# ---------- Cheia API (luată automat din .env prin config.py) ----------
api_key = GEMINI_API_KEY
 
with st.sidebar:
    st.header("Setări")
    if api_key:
        st.success("✅ Cheie API încărcată automat din .env.")
    else:
        st.error(
            "❌ Nu am găsit GEMINI_API_KEY.\n\n"
            "Adaugă-l în fișierul `.env`:\n"
            '`GEMINI_API_KEY=cheia_ta_aici`'
        )
        st.caption("Obține gratuit o cheie pe: https://aistudio.google.com/apikey")
    model = st.selectbox("Model", ["gemini-2.0-flash", "gemini-2.0-flash-lite"], index=0)
 
# ---------- State ----------
if "context" not in st.session_state:
    st.session_state.context = ""
if "url_curent" not in st.session_state:
    st.session_state.url_curent = ""
if "istoric" not in st.session_state:
    st.session_state.istoric = []
 
 
def extrage_text_din_url(url: str) -> str:
    resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
 
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()
 
    text = soup.get_text(separator="\n")
    linii = [linie.strip() for linie in text.splitlines() if linie.strip()]
    return "\n".join(linii)
 
 
def raspunde(api_key: str, model_name: str, context: str, intrebare: str, istoric: list) -> str:
    mesaje_istoric = ""
    for q, a in istoric[-3:]:
        mesaje_istoric += f"Întrebare anterioară: {q}\nRăspuns anterior: {a}\n\n"
 
    prompt = f"""Ai la dispoziție următorul conținut extras de pe un site:
 
---
{context}
---
 
{mesaje_istoric}
Pe baza EXCLUSIV a conținutului de mai sus, răspunde la întrebarea de mai jos.
Dacă informația nu se găsește în conținut, spune clar că nu ai găsit răspunsul pe pagină.
 
Întrebare: {intrebare}
"""
 
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    response = model.generate_content(prompt)
    return response.text
 
 
# ---------- UI: link ----------
url = st.text_input("🔗 Link către pagina HTML", placeholder="https://exemplu.ro/pagina")
 
col1, col2 = st.columns([1, 3])
with col1:
    incarca = st.button("Încarcă pagina", use_container_width=True)
 
if incarca:
    if not url:
        st.warning("Introdu un link mai întâi.")
    else:
        with st.spinner("Se extrage conținutul paginii..."):
            try:
                st.session_state.context = extrage_text_din_url(url)
                st.session_state.url_curent = url
                st.session_state.istoric = []
                st.success(f"Pagina a fost încărcată ({len(st.session_state.context)} caractere de text extras).")
            except Exception as e:
                st.error(f"Eroare la încărcarea paginii: {e}")
 
if st.session_state.context:
    with st.expander("📄 Vezi conținutul extras din pagină"):
        st.text(st.session_state.context[:5000] + ("..." if len(st.session_state.context) > 5000 else ""))
 
st.divider()
 
# ---------- UI: întrebări ----------
st.subheader("💬 Întreabă agentul")
 
intrebare = st.text_input("Întrebarea ta", placeholder="Ex: Care sunt orele de program?")
intreaba_btn = st.button("Trimite întrebarea")
 
if intreaba_btn:
    if not api_key:
        st.warning("Cheia API lipsește din .env — vezi mesajul din bara laterală.")
    elif not st.session_state.context:
        st.warning("Încarcă mai întâi o pagină web.")
    elif not intrebare:
        st.warning("Scrie o întrebare.")
    else:
        with st.spinner("Agentul gândește..."):
            try:
                raspuns = raspunde(api_key, model, st.session_state.context, intrebare, st.session_state.istoric)
                st.session_state.istoric.append((intrebare, raspuns))
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
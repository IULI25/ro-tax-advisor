import streamlit as st
import requests
import google.generativeai as genai
 

 
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
 
st.set_page_config(page_title="Agent AI din pagină web", page_icon="🤖", layout="centered")
st.title("🤖 Consilier AI")
 
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
if "chunkuri" not in st.session_state:
    st.session_state.chunkuri = []
if "istoric" not in st.session_state:
    st.session_state.istoric = []
 
 
 
def raspunde(api_key: str, model_name: str, chunkuri: list, intrebare: str, istoric: list) -> str:
    # selectăm doar fragmentele relevante pentru întrebare, nu tot documentul
    chunkuri_relevante = selecteaza_chunkuri_relevante(chunkuri, intrebare, top_k=5)
    context = "\n\n---\n\n".join(c["text"] for c in chunkuri_relevante)
 
    mesaje_istoric = ""
    for q, a in istoric[-3:]:
        mesaje_istoric += f"Întrebare anterioară: {q}\nRăspuns anterior: {a}\n\n"
 
    prompt = f"""Ai la dispoziție următoarele fragmente extrase de pe un site:
 
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
 
 
# ---------- UI: întrebări ----------
st.subheader("💬 Întreabă agentul")
 
intrebare = st.text_input("Întrebarea ta", placeholder="Ex: Care sunt orele de program?")
intreaba_btn = st.button("Trimite întrebarea")
 
if intreaba_btn:
    if not api_key:
        st.warning("Cheia API lipsește din .env — vezi mesajul din bara laterală.")
    elif not st.session_state.chunkuri:
        st.warning("Încarcă mai întâi o pagină web.")
    elif not intrebare:
        st.warning("Scrie o întrebare.")
    else:
        with st.spinner("Agentul gândește..."):
            try:
                raspuns = raspunde(api_key, model, st.session_state.chunkuri, intrebare, st.session_state.istoric)
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
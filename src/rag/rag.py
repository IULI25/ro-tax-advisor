import streamlit as st
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
import os
from pathlib import Path
from docling.document_converter import DocumentConverter
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

st.set_page_config(page_title="Agent AI din pagină web", page_icon="🤖", layout="centered")
 
st.title("🤖 Consilier AI ")

 
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
if "istoric" not in st.session_state:
    st.session_state.istoric = []
 
 
def extrage_text_din_fisier(file_path: str, save_html: bool = True, save_json: bool = True) -> str:
    path_obj = Path(file_path).resolve()
    
    # 1. Verify file exists locally before running Docling
    if not path_obj.is_file():
        raise FileNotFoundError(f"Fisierul nu a fost gasit la cale: {path_obj}")

    # 2. Setup directories
    html_dir = Path("tmp/html")
    docling_dir = Path("tmp/docling")
    html_dir.mkdir(parents=True, exist_ok=True)
    docling_dir.mkdir(parents=True, exist_ok=True)

    # 3. Convert via Docling (passing Path object)
    converter = DocumentConverter()
    result = converter.convert(path_obj)
    docling_doc = result.document

    # 4. Save visual HTML check
    if save_html:
        html_out = html_dir / f"{path_obj.stem}.html"
        with open(html_out, "w", encoding="utf-8") as f:
            f.write(docling_doc.export_to_html())

    # 5. Save JSON for reuse
    if save_json:
        json_out = docling_dir / f"{path_obj.stem}.json"
        with open(json_out, "w", encoding="utf-8") as f:
            f.write(docling_doc.model_dump_json())

    return docling_doc.export_to_markdown()

# Example usage:
text = extrage_text_din_fisier("/ro-tax-advisor/src/ingestion/Legea nr.227_2015.html")

def raspunde(api_key: str, model_name: str, docling_doc: str, intrebare: str, istoric: list) -> str:
    mesaje_istoric = ""
    for q, a in istoric[-3:]:
        mesaje_istoric += f"Întrebare anterioară: {q}\nRăspuns anterior: {a}\n\n"
 
    prompt = f"""Ai la dispoziție următorul conținut extras de pe un site:
 
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
    elif not docling_doc.export_to_markdown:
        st.warning("Încarcă mai întâi o pagină web.")
    elif not intrebare:
        st.warning("Scrie o întrebare.")
    else:
        with st.spinner("Agentul gândește..."):
            try:
                raspuns = raspunde(api_key, model, docling_doc.export_to_markdown, intrebare, st.session_state.istoric)
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
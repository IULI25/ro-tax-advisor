import streamlit as st
import requests
from bs4 import BeautifulSoup
from openai import OpenAI

st.set_page_config(page_title="Agent AI din pagină web", page_icon="🤖", layout="centered")

st.title("🤖 Agent AI - răspunde din conținutul unei pagini web")
st.write(
    "Introdu un link, apoi pune întrebări. Agentul va răspunde "
    "folosind exclusiv conținutul extras de pe acea pagină."
)

# ---------- Sidebar: cheia API ----------
with st.sidebar:
    st.header("Setări")
    api_key = st.text_input("OpenAI API Key", type="password")
    model = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o"], index=0)
    st.caption("Cheia nu este salvată nicăieri, se folosește doar pentru sesiunea curentă.")

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


def raspunde(client: OpenAI, model: str, context: str, intrebare: str, istoric: list) -> str:
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

    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return completion.choices[0].message.content


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
        st.warning("Introdu cheia OpenAI API în bara laterală.")
    elif not st.session_state.context:
        st.warning("Încarcă mai întâi o pagină web.")
    elif not intrebare:
        st.warning("Scrie o întrebare.")
    else:
        with st.spinner("Agentul gândește..."):
            try:
                client = OpenAI(api_key=api_key)
                raspuns = raspunde(client, model, st.session_state.context, intrebare, st.session_state.istoric)
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

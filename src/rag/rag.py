"""
Streamlit Chat Interface — OpenAI + simple RAG (Retrieval-Augmented Generation)

This app answers questions using YOUR documents as the "direct information source",
instead of relying only on the model's training data.

How it works:
1. Put your source documents (.txt or .pdf) in a folder called "knowledge_base/"
2. On first run, the app chunks the documents and embeds them with OpenAI embeddings
3. When you ask a question, it finds the most relevant chunks and feeds them to
   the model as context, so answers are grounded in your actual documents.

Setup:
    pip install streamlit openai numpy pypdf

Run:
    streamlit run chat_app_openai_rag.py
"""

import os
import glob
import numpy as np
import streamlit as st
from openai import OpenAI

# ---------- Page config ----------
st.set_page_config(page_title="Chat", page_icon="💬", layout="centered")

KNOWLEDGE_DIR = "knowledge_base"
EMBED_MODEL = "text-embedding-3-small"
CHUNK_SIZE = 800       # characters per chunk
CHUNK_OVERLAP = 100
TOP_K = 4              # how many chunks to retrieve per question

# ---------- Sidebar ----------
with st.sidebar:
    st.title("⚙️ Settings")

    api_key = st.text_input(
        "OpenAI API Key",
        value=os.environ.get("OPENAI_API_KEY", ""),
        type="password",
        help="Get a key at https://platform.openai.com/api-keys",
    )

    model = st.selectbox(
        "Chat model",
        ["gpt-4.1", "gpt-4.1-mini", "gpt-4o", "gpt-4o-mini"],
        index=1,
    )

    system_prompt = st.text_area(
        "System prompt",
        value=(
            "You are a helpful assistant. Answer using ONLY the provided "
            "context when it's relevant. If the context doesn't contain the "
            "answer, say so clearly instead of guessing."
        ),
        height=100,
    )

    max_tokens = st.slider("Max tokens", 256, 4096, 1024, step=256)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.3, step=0.1)

    st.divider()
    st.caption(f"📁 Knowledge base folder: `{KNOWLEDGE_DIR}/`")

    if st.button("🔄 Rebuild index", use_container_width=True):
        st.session_state.pop("kb_index", None)
        st.rerun()

    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ---------- Session state ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------- Document loading & chunking ----------
def load_documents():
    os.makedirs(KNOWLEDGE_DIR, exist_ok=True)
    texts = []

    for path in glob.glob(os.path.join(KNOWLEDGE_DIR, "*.txt")):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            texts.append((os.path.basename(path), f.read()))

    for path in glob.glob(os.path.join(KNOWLEDGE_DIR, "*.pdf")):
        try:
            from pypdf import PdfReader
            reader = PdfReader(path)
            content = "\n".join(page.extract_text() or "" for page in reader.pages)
            texts.append((os.path.basename(path), content))
        except Exception as e:
            st.warning(f"Could not read {path}: {e}")

    return texts


def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return [c.strip() for c in chunks if c.strip()]


def build_index(client):
    docs = load_documents()
    if not docs:
        return None

    all_chunks = []
    for filename, content in docs:
        for chunk in chunk_text(content):
            all_chunks.append({"text": chunk, "source": filename})

    if not all_chunks:
        return None

    texts = [c["text"] for c in all_chunks]
    embeddings = []
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        embeddings.extend([d.embedding for d in resp.data])

    for chunk, emb in zip(all_chunks, embeddings):
        chunk["embedding"] = np.array(emb)

    return all_chunks


def retrieve(client, query, index, k=TOP_K):
    if not index:
        return []
    q_emb = np.array(
        client.embeddings.create(model=EMBED_MODEL, input=[query]).data[0].embedding
    )
    scored = []
    for chunk in index:
        sim = np.dot(q_emb, chunk["embedding"]) / (
            np.linalg.norm(q_emb) * np.linalg.norm(chunk["embedding"]) + 1e-8
        )
        scored.append((sim, chunk))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:k]]


# ---------- Header ----------
st.title("💬 Chat with your documents")
st.caption(f"Model: `{model}` · Retrieval-augmented with files in `{KNOWLEDGE_DIR}/`")

if not api_key:
    st.info("Enter your OpenAI API key in the sidebar to get started.")
    st.stop()

client = OpenAI(api_key=api_key)

# Build / load the retrieval index once per session
if "kb_index" not in st.session_state:
    with st.spinner("Indexing knowledge base..."):
        st.session_state.kb_index = build_index(client)

index = st.session_state.kb_index
if index:
    st.caption(f"✅ Indexed {len(index)} chunks from your documents.")
else:
    st.caption("⚠️ No documents found — add .txt or .pdf files to the knowledge_base/ folder.")

# ---------- Render chat history ----------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------- Chat input ----------
prompt = st.chat_input("Ask a question...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Retrieve relevant context
    retrieved = retrieve(client, prompt, index)
    context_block = "\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}" for c in retrieved
    )

    grounded_system_prompt = system_prompt
    if context_block:
        grounded_system_prompt += f"\n\nContext from documents:\n{context_block}"

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        try:
            stream = client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
                messages=[
                    {"role": "system", "content": grounded_system_prompt},
                    *[
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages
                    ],
                ],
            )
            for event in stream:
                delta = event.choices[0].delta.content or ""
                full_response += delta
                placeholder.markdown(full_response + "▌")
            placeholder.markdown(full_response)

            if retrieved:
                with st.expander("📎 Sources used"):
                    for c in retrieved:
                        st.markdown(f"**{c['source']}**")
                        st.caption(c["text"][:300] + "...")

        except Exception as e:
            full_response = f"⚠️ Error: {e}"
            placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
"""
Streamlit Chat Interface for Claude (Anthropic API)

Setup:
    pip install streamlit anthropic

Run:
    streamlit run chat_app.py

You'll be prompted for your Anthropic API key in the sidebar,
or you can set it as an environment variable: ANTHROPIC_API_KEY
"""

import os
import streamlit as st
from anthropic import Anthropic

# ---------- Page config ----------
st.set_page_config(page_title="Chat", page_icon="💬", layout="centered")

# ---------- Sidebar ----------
with st.sidebar:
    st.title("⚙️ Settings")

    api_key = st.text_input(
        "Anthropic API Key",
        value=os.environ.get("ANTHROPIC_API_KEY", ""),
        type="password",
        help="Get a key at https://console.anthropic.com",
    )

    model = st.selectbox(
        "Model",
        [
            "claude-sonnet-4-6",
            "claude-opus-4-6",
            "claude-haiku-4-6",
        ],
        index=0,
    )

    system_prompt = st.text_area(
        "System prompt",
        value="You are a helpful, friendly assistant.",
        height=100,
    )

    max_tokens = st.slider("Max tokens", 256, 4096, 1024, step=256)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, step=0.1)

    st.divider()
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ---------- Session state ----------
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role": ..., "content": ...}

# ---------- Header ----------
st.title("💬 Chat with Claude")
st.caption(f"Model: `{model}`")

# ---------- Render chat history ----------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------- Chat input ----------
prompt = st.chat_input("Type a message...")

if prompt:
    if not api_key:
        st.error("Please enter your Anthropic API key in the sidebar.")
        st.stop()

    # Show and store the user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call the API and stream the response
    client = Anthropic(api_key=api_key)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        try:
            with client.messages.stream(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
            ) as stream:
                for text in stream.text_stream:
                    full_response += text
                    placeholder.markdown(full_response + "▌")
            placeholder.markdown(full_response)
        except Exception as e:
            full_response = f"⚠️ Error: {e}"
            placeholder.markdown(full_response)

    st.session_state.messages.append(
        {"role": "assistant", "content": full_response}
    )
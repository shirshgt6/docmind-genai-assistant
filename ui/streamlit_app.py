"""Streamlit chat UI for DocMind.  Run:  streamlit run ui/streamlit_app.py"""
import os
import sys
import uuid
from pathlib import Path

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")
# "embedded" runs the FastAPI app in-process (single-process hosts like Streamlit Community Cloud).
EMBEDDED = os.getenv("DOCMIND_MODE", "").lower() == "embedded"

st.set_page_config(page_title="DocMind", page_icon="📄", layout="wide")
st.title("📄 DocMind — chat with your documents")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []


@st.cache_resource
def embedded_client():
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from fastapi.testclient import TestClient

    from docmind.api import app

    return TestClient(app, raise_server_exceptions=False)


def api(method: str, path: str, **kwargs):
    try:
        if EMBEDDED:
            res = embedded_client().request(method, path, **kwargs)
        else:
            res = requests.request(method, f"{API_URL}{path}", timeout=120, **kwargs)
    except requests.ConnectionError:
        st.error(f"Backend not reachable at {API_URL}. Start it with: uvicorn docmind.api:app --reload")
        st.stop()
    if res.status_code >= 400:
        try:
            st.error(res.json().get("detail", res.text))
        except ValueError:
            st.error(f"Backend error {res.status_code}. Check the uvicorn terminal for the traceback.")
        return None
    return res.json() if res.content else {}


# ---------- sidebar: documents ----------
with st.sidebar:
    st.header("Documents")
    uploaded = st.file_uploader("Upload PDF / TXT / MD", type=["pdf", "txt", "md"])
    if uploaded and st.button("Ingest file"):
        with st.spinner("Chunking + embedding..."):
            out = api("POST", "/documents/upload", files={"file": (uploaded.name, uploaded.getvalue())})
        if out:
            st.success(f"Added {out['chunks_added']} chunks from {', '.join(out['sources'])}")

    url = st.text_input("...or a web page URL")
    if url and st.button("Ingest URL"):
        with st.spinner("Fetching page..."):
            out = api("POST", "/documents/url", json={"url": url})
        if out:
            st.success(f"Added {out['chunks_added']} chunks")

    sources = (api("GET", "/documents") or {}).get("sources", [])
    st.caption("Indexed: " + (", ".join(sources) if sources else "nothing yet"))

    st.divider()
    mode = st.radio("Mode", ["Smart assistant (router)", "RAG Q&A", "Agent (tools)"])
    if st.button("New chat"):
        api("DELETE", f"/sessions/{st.session_state.session_id}")
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []

# ---------- tabs ----------
chat_tab, insights_tab = st.tabs(["💬 Chat", "🔎 Summary & insights"])

with chat_tab:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if question := st.chat_input("Ask about your documents..."):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        body = {"question": question, "session_id": st.session_state.session_id}
        with st.chat_message("assistant"), st.spinner("Thinking..."):
            if mode == "RAG Q&A":
                out = api("POST", "/ask", json=body)
                answer = out and out["answer"]
                if out and out["sources"]:
                    with st.expander("Sources"):
                        for s in out["sources"]:
                            page = f" (page {s['page'] + 1})" if s["page"] is not None else ""
                            st.markdown(f"**{s['chunk_id']}**{page} — {s['snippet']}...")
            elif mode == "Agent (tools)":
                out = api("POST", "/agent", json=body)
                answer = out and out["answer"]
                if out and out["steps"]:
                    with st.expander("Tool calls"):
                        st.json(out["steps"])
            else:
                out = api("POST", "/assistant", json=body)
                if out:
                    st.caption(f"Routed to: **{out['intent']}**")
                    result = out["result"]
                    answer = result.get("answer") or result.get("summary") or f"```json\n{result}\n```"
                else:
                    answer = None
            if answer:
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

with insights_tab:
    source = st.selectbox("Document", ["All documents", *sources])
    chosen = None if source == "All documents" else source
    col1, col2 = st.columns(2)
    with col1:
        style = st.selectbox("Summary style", ["concise bullet-point", "executive summary", "explain like I'm 5"])
        if st.button("Summarize"):
            with st.spinner("Summarizing..."):
                out = api("POST", "/summarize", json={"source": chosen, "style": style})
            if out:
                st.markdown(out["summary"])
    with col2:
        if st.button("Extract insights"):
            with st.spinner("Extracting..."):
                out = api("POST", "/extract", json={"source": chosen})
            if out:
                st.subheader(out["title"])
                st.write(f"**Type:** {out['document_type']} · **Sentiment:** {out['sentiment']}")
                st.write("**Key points**")
                for p in out["key_points"]:
                    st.markdown(f"- {p}")
                if out["entities"]:
                    st.write("**Entities:** " + ", ".join(out["entities"]))

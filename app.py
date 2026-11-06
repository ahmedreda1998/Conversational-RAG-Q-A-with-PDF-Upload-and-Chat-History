"""

Key Streamlit fact: the whole script RE-RUNS from top to bottom on every
interaction. Anything expensive (embedding model, vector store, history
store) must therefore live in `st.session_state` (per user) or behind
`@st.cache_resource` (shared), otherwise you'll reload the model on every
click and lose all chat memory.

Spec security rule: the API key lives in st.session_state only -- use
st.text_input(type="password"), never write it to disk or print it.

The UI must only talk to the `Chatbot` façade, never to LangChain.
"""
import os

import streamlit as st

from config import Config
from chatbot import Chatbot
from engines.direct_llm_engine import DirectLlmEngine
from engines.rag_engine import RagEngine
from rag.ingest import PdfIngestor, VectorStoreClient
from rag.pipeline import LLMClient, RagPipelineBuilder, SessionHistoryStore


def init_state() -> None:
    """
    Set defaults in st.session_state if missing, e.g.
      "api_key", "chatbot", "ingestor", "history_store", "ingested_files".
    Pattern:  if "x" not in st.session_state: st.session_state.x = ...
    """
    if "history_store" not in st.session_state:
        st.session_state.history_store = SessionHistoryStore()
    if "direct_history_store" not in st.session_state:
        st.session_state.direct_history_store = SessionHistoryStore()

    if "ingested_files" not in st.session_state:
        st.session_state.ingested_files = []
    


@st.cache_resource
def get_vector_store(_config: Config) -> VectorStoreClient:
    """
    Build ONE VectorStoreClient for the whole app (creates the embeddings
    client once). Embeddings run locally on Ollama, so the API key is not
    needed here and is not part of the cache key. The leading underscore on
    `_config` tells Streamlit not to hash that argument.
    """
    return VectorStoreClient(_config, api_key="")


def build_chatbot(config: Config, api_key: str) -> Chatbot:
    """
    Composition root: the ONLY place where concrete classes are chosen.
      store   = get_vector_store(config, api_key)
      history = SessionHistoryStore()   (keep it in session_state so it
                                         survives reruns!)
      llm     = LLMClient(config, api_key)
      chain   = RagPipelineBuilder(config, store, llm, history).build_with_history()
      return Chatbot(RagEngine(chain, history, store))

    To fall back to the legacy bot, return Chatbot(SimpleFAQEngine({...})).
    """

    store = get_vector_store(config)
    history = st.session_state.history_store
    llm = LLMClient(config, api_key)
    chain = RagPipelineBuilder(config, store, llm, history).build_with_history()
    return Chatbot(RagEngine(chain, history, store))


def build_direct_chatbot(config: Config, api_key: str) -> Chatbot:
    """Plain LLM chat (no PDFs). Own history store, so it never mixes with RAG chats."""
    llm = LLMClient(config, api_key).build()
    return Chatbot(DirectLlmEngine(llm, st.session_state.direct_history_store))


MODE_RAG = "Chat with my PDFs (RAG)"
MODE_DIRECT = "Ask the model directly"


def render_sidebar(config: Config) -> tuple:
    """
    Draw the sidebar: mode switch, session ID and, in RAG mode only, the PDF
    uploader with a "Process PDFs" button that ingests the files and reports
    per-file success/failure. Returns (mode, session_id).
    """

    st.sidebar.title("Settings")
    # Held in session_state only (widget key), never written to disk.
    st.sidebar.text_input(
        "API Key", type="password", key="api_key_input",
        help="Optional if OPENAI_API_KEY is set in .env. Kept in memory only.",
    )
    mode = st.sidebar.radio("Mode", [MODE_RAG, MODE_DIRECT])
    session_id = st.sidebar.text_input("Session ID", value="default")
    if mode == MODE_DIRECT:
        st.sidebar.caption("The model answers from its own knowledge. No PDFs are used.")
        return mode, session_id
    uploaded_files = st.sidebar.file_uploader(
        "Upload PDFs", type=["pdf"], accept_multiple_files=True
    )
    if st.sidebar.button("Process PDFs"):
        if not uploaded_files:
            st.sidebar.error("No files uploaded.")
        else:
            # Same cached store the chatbot uses, so both see the same data.
            store = get_vector_store(config)
            ingestor = PdfIngestor(config, store)
            with st.spinner("Processing PDFs..."):
                summary = ingestor.ingest_uploaded_files(uploaded_files)
            for name, n_chunks in summary["ingested"].items():
                st.sidebar.success(f"{name}: {n_chunks} chunks")
                if name not in st.session_state.ingested_files:
                    st.session_state.ingested_files.append(name)
            for name, reason in summary["failed"].items():
                st.sidebar.error(f"{name}: {reason}")
    if st.session_state.ingested_files:
        st.sidebar.markdown("**Processed files**")
        for name in st.session_state.ingested_files:
            st.sidebar.write(f"- {name}")
    return mode, session_id


def render_transcript(chatbot: Chatbot, session_id: str) -> None:
    """
    Draw every stored message of this session (the spec's visible chat
    history).
    """
    # The n-th assistant message belongs to the n-th entry of chatbot.sources().
    sources = chatbot.sources(session_id)
    answer_index = 0
    for role, text in chatbot.history(session_id):
        with st.chat_message(role):
            st.write(text)
            if role == "assistant":
                if answer_index < len(sources):
                    render_sources(sources[answer_index])
                answer_index += 1


def render_sources(sources: list) -> None:
    """Show the PDFs/pages an answer was based on, inside a collapsed expander."""
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for s in sources:
            page = f", page {s['page']}" if s["page"] is not None else ""
            st.markdown(f"**{s['source']}{page}**")
            st.caption(s["snippet"])


def handle_user_input(chatbot: Chatbot, session_id: str) -> None:
    """
    Read a question, echo it, then stream the answer and show its sources.
    An empty vector store is handled inside the engine, which replies with an
    "upload a PDF" message (that reply is not saved, so it has no sources).
    """
    question = st.chat_input("Ask about your PDFs...")
    if not question:
        return
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        turns_before = len(chatbot.sources(session_id))
        st.write_stream(chatbot.ask_stream(session_id, question))
        turns = chatbot.sources(session_id)
        if len(turns) > turns_before:  # this turn was answered and recorded
            render_sources(turns[-1])


def main() -> None:
    """
    Page setup, sidebar, then (once an API key exists) build the chatbot,
    rebuilding it only when the key changes, and draw transcript + input.
    """
    st.set_page_config(page_title="Conversational RAG", layout="wide")
    st.title("Conversational RAG")
    init_state()
    config = Config()
    mode, session_id = render_sidebar(config)
    # Key typed in the sidebar wins; .env is only a fallback.
    api_key = (st.session_state.get("api_key_input", "").strip()
               or os.getenv("OPENAI_API_KEY", "").strip())
    if not api_key:
        st.error("Enter your API key in the sidebar (or set OPENAI_API_KEY in .env).")
        return
    # Rebuild both chatbots when the key changes. Chat histories live in
    # separate stores, so the transcripts survive a rebuild.
    if st.session_state.get("built_with_key") != api_key:
        st.session_state.pop("chatbot", None)
        st.session_state.pop("direct_chatbot", None)
        st.session_state.built_with_key = api_key
    # Each mode has its own chatbot, built once and kept across reruns.
    if mode == MODE_DIRECT:
        if "direct_chatbot" not in st.session_state:
            st.session_state.direct_chatbot = build_direct_chatbot(config, api_key)
        chatbot = st.session_state.direct_chatbot
    else:
        if "chatbot" not in st.session_state:
            st.session_state.chatbot = build_chatbot(config, api_key)
        chatbot = st.session_state.chatbot
    render_transcript(chatbot, session_id)
    handle_user_input(chatbot, session_id)


if __name__ == "__main__":
    main()

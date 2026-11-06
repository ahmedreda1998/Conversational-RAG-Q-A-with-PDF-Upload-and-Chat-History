"""
RagEngine is the "adapter" between LangChain's world and ChatEngine
interface.

Extras beyond the base contract:
  - answer_stream(): same as answer() but yields the text piece by piece.
  - get_sources():   the PDF name / page / snippet used for each answer.
"""
from typing import Dict, Iterator, List, Tuple

from openai import AuthenticationError, BadRequestError, NotFoundError, RateLimitError

from engines.base import ChatEngine
from rag.ingest import VectorStoreClient
from rag.pipeline import SessionHistoryStore

EMPTY_STORE_MESSAGE = "The vector store is empty. Please upload a PDF to continue."
SNIPPET_CHARS = 200


def friendly_error(e: Exception) -> str:
    """Map an exception to a message safe to show in the UI."""
    if isinstance(e, AuthenticationError):
        return "Authentication failed. Please check your OpenAI API key."
    if isinstance(e, RateLimitError):
        return "Rate limit or quota reached. Please wait a moment and try again."
    if isinstance(e, NotFoundError):
        return "The configured model was not found for your account. Check llm_model in config.py."
    if isinstance(e, BadRequestError):
        return "The request was rejected by OpenAI (for example an unsupported setting such as temperature)."
    # Only the exception type is shown: the message text may echo the API key.
    return f"Sorry, something went wrong ({type(e).__name__}). Please try again."


def summarize_sources(docs: list) -> List[Dict]:
    """
    Turn retrieved Documents into plain dicts the UI can show without knowing
    LangChain: {"source": filename, "page": 1-based int or None, "snippet": str}.
    Several chunks from the same page are collapsed into one entry.
    """
    seen = set()
    result = []
    for doc in docs:
        meta = doc.metadata or {}
        source = meta.get("source", "unknown")
        page = meta.get("page")
        page = page + 1 if isinstance(page, int) else None  # PyPDFLoader pages are 0-based
        if (source, page) in seen:
            continue
        seen.add((source, page))
        result.append({
            "source": source,
            "page": page,
            "snippet": doc.page_content[:SNIPPET_CHARS],
        })
    return result


class RagEngine(ChatEngine):
    def __init__(self, rag_chain_with_history, history_store: SessionHistoryStore,
                 vector_store: VectorStoreClient = None):
        """
        store the chain, the history store, and (optionally) the vector
        store so answer() can check for an empty store.

        self.sources[session_id] is a list with one entry per ANSWERED turn
        (same order as the assistant messages in the history), each a list of
        source dicts. Turns that failed or hit the empty store add nothing,
        because they are not saved in the history either.
        """
        self.chain = rag_chain_with_history
        self.history_store = history_store
        self.vector_store = vector_store
        self.sources: Dict[str, List[List[Dict]]] = {}

    def _is_empty(self) -> bool:
        return self.vector_store is not None and self.vector_store.is_empty()

    def _record_sources(self, session_id: str, docs: list) -> None:
        self.sources.setdefault(session_id, []).append(summarize_sources(docs))

    def answer(self, session_id: str, question: str) -> str:
        """
        0. If the vector store is empty, return a friendly message asking
           the user to upload a PDF (don't call the LLM).
        1. Invoke the chain with {"input": question} for this session.
        2. Remember the retrieved chunks as sources for the UI.
        3. Return resp["answer"]. Errors become a friendly message.
        """
        if self._is_empty():
            return EMPTY_STORE_MESSAGE

        try:
            resp = self.chain.invoke(
                {"input": question},
                config={"configurable": {"session_id": session_id}},
            )
            self._record_sources(session_id, resp.get("context", []))
            return resp.get("answer", "No answer returned.")
        except Exception as e:
            return friendly_error(e)

    def answer_stream(self, session_id: str, question: str) -> Iterator[str]:
        """
        Like answer(), but yields the reply in small text pieces.

        The chain streams partial dicts: "context" arrives once (the retrieved
        Documents), then "answer" arrives in many small strings. The history
        is saved by the chain only after the stream is fully consumed, and
        sources are recorded at the same moment, so the two stay aligned.
        If something fails midway the error text is yielded at the end and
        the turn is not saved.
        """
        if self._is_empty():
            yield EMPTY_STORE_MESSAGE
            return

        context = []
        try:
            for chunk in self.chain.stream(
                {"input": question},
                config={"configurable": {"session_id": session_id}},
            ):
                if "context" in chunk:
                    context = chunk["context"]
                piece = chunk.get("answer")
                if piece:
                    yield piece
            self._record_sources(session_id, context)
        except Exception as e:
            yield "\n\n" + friendly_error(e)

    def get_history(self, session_id: str) -> List[Tuple[str, str]]:
        """
        Read the session's ChatMessageHistory from the history store.
        If it doesn't exist return []. Otherwise convert each message:
            m.type == "human" -> ("user", m.content)
            else              -> ("assistant", m.content)
        """
        history = self.history_store.peek(session_id)
        if history is None:
            return []
        return [("user", m.content) if m.type == "human" else ("assistant", m.content)
                for m in history.messages]

    def get_sources(self, session_id: str) -> List[List[Dict]]:
        """One list of source dicts per answered turn, oldest first (a copy)."""
        return [list(turn) for turn in self.sources.get(session_id, [])]

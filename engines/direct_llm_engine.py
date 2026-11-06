"""
engines/direct_llm_engine.py  --  plain chat with the LLM, no PDFs, no retrieval.

Same ChatEngine contract as RagEngine, so the UI can swap between them. It
still keeps a per-session history, so follow-up questions work.
"""
from typing import Iterator, List, Tuple

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

from engines.base import ChatEngine
from engines.rag_engine import friendly_error
from rag.pipeline import SessionHistoryStore

DIRECT_SYSTEM_PROMPT = "You are a helpful assistant. Answer clearly and concisely."


class DirectLlmEngine(ChatEngine):
    def __init__(self, llm, history_store: SessionHistoryStore):
        """`llm` is a LangChain chat model, e.g. LLMClient(...).build()."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", DIRECT_SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        self.history_store = history_store
        self.chain = RunnableWithMessageHistory(
            prompt | llm,
            history_store.get,
            input_messages_key="input",
            history_messages_key="chat_history",
        )

    def _config(self, session_id: str) -> dict:
        return {"configurable": {"session_id": session_id}}

    def answer(self, session_id: str, question: str) -> str:
        try:
            reply = self.chain.invoke({"input": question}, config=self._config(session_id))
            return reply.content
        except Exception as e:
            return friendly_error(e)

    def answer_stream(self, session_id: str, question: str) -> Iterator[str]:
        """Yield the reply in pieces. On failure the friendly error is yielded last."""
        try:
            for chunk in self.chain.stream({"input": question}, config=self._config(session_id)):
                if chunk.content:
                    yield chunk.content
        except Exception as e:
            yield "\n\n" + friendly_error(e)

    def get_history(self, session_id: str) -> List[Tuple[str, str]]:
        history = self.history_store.peek(session_id)
        if history is None:
            return []
        return [("user", m.content) if m.type == "human" else ("assistant", m.content)
                for m in history.messages]

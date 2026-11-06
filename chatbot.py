from typing import Dict, Iterator, List, Tuple

from engines.base import ChatEngine


class Chatbot:
    def __init__(self, engine: ChatEngine):
        self.engine = engine

    def ask(self, session_id: str, question: str) -> str:
        return self.engine.answer(session_id, question)

    def ask_stream(self, session_id: str, question: str) -> Iterator[str]:
        """Yield the answer in pieces. Engines without streaming yield it whole."""
        stream = getattr(self.engine, "answer_stream", None)
        if stream is None:
            yield self.engine.answer(session_id, question)
        else:
            yield from stream(session_id, question)

    def history(self, session_id: str) -> List[Tuple[str, str]]:
        return self.engine.get_history(session_id)

    def sources(self, session_id: str) -> List[List[Dict]]:
        """Sources per answered turn (oldest first). Empty if the engine has none."""
        get_sources = getattr(self.engine, "get_sources", None)
        return get_sources(session_id) if get_sources else []

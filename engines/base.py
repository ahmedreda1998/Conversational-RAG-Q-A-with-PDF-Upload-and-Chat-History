from typing import Protocol, List, Tuple


class ChatEngine(Protocol):
    def answer(self, session_id: str, question: str) -> str: ...

    def get_history(self, session_id: str) -> List[Tuple[str, str]]: ...

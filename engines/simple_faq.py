from typing import Dict, List, Tuple

from engines.base import ChatEngine


class SimpleFAQEngine(ChatEngine):
    def __init__(self, faq_map: Dict[str, str]):

        self.faq_map = faq_map
        self.histories = {}

    def answer(self, session_id: str, question: str) -> str:
        ans = self.faq_map.get(question.strip().lower(), "Sorry, I don't know.")
        history = self.histories.setdefault(session_id, [])
        history.append(("user", question))
        history.append(("assistant", ans))
        return ans


    def get_history(self, session_id: str) -> List[Tuple[str, str]]:
        history = list(self.histories.get(session_id, []))
        return history
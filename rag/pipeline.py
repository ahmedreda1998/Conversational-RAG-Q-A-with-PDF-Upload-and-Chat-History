"""
Conversational RAG has two LLM steps per turn:

    follow-up question + chat history
        |
        v
 [1] CONTEXTUALIZE: LLM rewrites it as a standalone question
     ("What about its price?" -> "What is the price of the X-200 laptop?")
        |
        v
 [2] RETRIEVE: standalone question -> Chroma -> top-k chunks
        |
        v
 [3] ANSWER: LLM gets (chunks + history + question) and must answer ONLY
     from the chunks.

Why step 1? The vector search has no idea what "its" means. Without
rewriting, follow-ups retrieve garbage.

"""
from typing import Dict

from langchain_openai import ChatOpenAI

from config import Config
from rag.ingest import VectorStoreClient

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain


# ---------------------------------------------------------------- prompts
# prompts are just strings. Keep them as module constants so you
# can tweak wording without touching logic.

#   system prompt for the contextualize step. It should say:
#   "Given the chat history and the latest user question, which might
#    reference the history, rewrite it as a standalone question.
#    Do NOT answer it. Return it unchanged if no rewrite is needed."
CONTEXTUALIZE_SYSTEM_PROMPT = "Given the chat history and the latest user question, which might reference the history, rewrite it as a standalone question. Do NOT answer it. Return it unchanged if no rewrite is needed."

# system prompt for the answer step. It must contain a {context}
# placeholder (LangChain fills it with the retrieved chunks) and tell the
# model to: answer only from the context, say "I don't know" if it's not
# there, and keep answers concise.
QA_SYSTEM_PROMPT = (
    "You are an assistant for question-answering tasks. Answer ONLY from the "
    "retrieved context below. If the answer is not in the context, say that you "
    "don't know; do not use outside knowledge. Use three sentences maximum and "
    "keep the answer concise.\n\nContext:\n{context}"
)


class LLMClient:
    """
    Adapter that builds the chat model (spec: "LLMClient adapters to swap
    providers without UI changes").
    """

    def __init__(self, config: Config, api_key: str):
        """
        store config and the api key IN MEMORY ONLY.
        Never write it to a file, never print/log it (spec task 7).
        """
        self._config = config
        self._api_key = api_key


    def build(self):
        """
        return ChatOpenAI(api_key=..., model=config.llm_model,
        temperature=config.llm_temperature).
        Pass the key explicitly instead of os.environ so it isn't leaked
        into the process environment.
        """

        return ChatOpenAI(
            api_key=self._api_key,
            base_url=self._config.openai_base_url or None,
            model=self._config.llm_model,
            temperature=self._config.llm_temperature,
        )


class SessionHistoryStore:
    """
    RunnableWithMessageHistory needs a function
    `get_session_history(session_id) -> BaseChatMessageHistory`. This class
    owns the dict and provides that function. RagEngine also reads it to
    implement get_history().
    """

    def __init__(self):
        """self._store: Dict[str, ChatMessageHistory] = {}"""
        self._store: Dict[str, ChatMessageHistory] = {}

    def get(self, session_id: str):
        """
        return the ChatMessageHistory for this session, creating a
        new empty one the first time the session is seen.
        This exact method is what you pass to RunnableWithMessageHistory.
        """
        if session_id not in self._store:
            self._store[session_id] = ChatMessageHistory()
        return self._store[session_id]

    def peek(self, session_id: str):
        """
        return the history if it exists, else None -- WITHOUT creating
        one. Useful for get_history() so that merely viewing an unknown
        session doesn't create state.
        """
        return self._store.get(session_id)


class RagPipelineBuilder:
    """Assembles the full history-aware RAG chain."""

    def __init__(self, config: Config, vector_store: VectorStoreClient,
                 llm_client: LLMClient, history_store: SessionHistoryStore):
        """Store all four collaborators."""
        self.config = config
        self.vector_store = vector_store
        self.llm_client = llm_client
        self.history_store = history_store

    def build_contextualize_prompt(self):
        """
        ChatPromptTemplate.from_messages([
            ("system", CONTEXTUALIZE_SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        """

        return ChatPromptTemplate.from_messages([
            ("system", CONTEXTUALIZE_SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

    def build_qa_prompt(self):
        """
        same shape as above but with QA_SYSTEM_PROMPT (contains
        {context}).
        """
        return ChatPromptTemplate.from_messages([
            ("system", QA_SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

    def build_history_aware_retriever(self, llm):
        """
        create_history_aware_retriever(llm, retriever, contextualize_prompt)
        where retriever = vector_store.as_retriever(config.retriever_k).
        """
        retriever = self.vector_store.as_retriever(self.config.retriever_k)
        contextualize_prompt = self.build_contextualize_prompt()
        return create_history_aware_retriever(llm, retriever, contextualize_prompt)

    def build_rag_chain(self):
        """
          1. llm = llm_client.build()
          2. history_aware_retriever = ...
          3. qa_chain = create_stuff_documents_chain(llm, qa_prompt)
          4. return create_retrieval_chain(history_aware_retriever, qa_chain)
        The result's invoke() returns a dict with "input", "chat_history",
        "context" (the retrieved Documents -- great for citations) and
        "answer".
        """

        llm = self.llm_client.build()
        history_aware_retriever = self.build_history_aware_retriever(llm)
        qa_prompt = self.build_qa_prompt()
        qa_chain = create_stuff_documents_chain(llm, qa_prompt)
        return create_retrieval_chain(history_aware_retriever, qa_chain)

    def build_with_history(self):
        """
        Wrap build_rag_chain() with RunnableWithMessageHistory:
            RunnableWithMessageHistory(
                rag_chain,
                history_store.get,           # session_id -> history
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer",
            )
        This is what RagEngine receives (spec: `rag_chain_with_history`).
        The wrapper automatically loads history before the call and saves
        the new human/AI messages after it.
        """

        return RunnableWithMessageHistory(
            self.build_rag_chain(),
            self.history_store.get,           # session_id -> history
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

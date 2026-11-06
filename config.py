import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

# Reads the project's .env (OPENAI_API_KEY, OPENAI_BASE_URL) into the process.
load_dotenv()


@dataclass
class Config:
    # --- Vector store (spec task 3) ---
    persist_directory: str = ".chroma/student-rag"
    collection_name: str = "student-rag"

    # --- Chunking (spec task 2) ---
    chunk_size: int = 1000
    chunk_overlap: int = 150

    # --- Embeddings ---
    embedding_model: str = "hf.co/mixedbread-ai/mxbai-embed-large-v1:F16"  # served locally by Ollama
    ollama_base_url: str = "http://localhost:11434"

    # --- Retrieval ---
    retriever_k: int = 4

    # --- LLM ---
    # Endpoint comes from .env (OPENAI_BASE_URL); the key is read in app.py
    # and deliberately NOT stored here so it never shows up in a repr/log.
    openai_base_url: str = field(default_factory=lambda: os.getenv("OPENAI_BASE_URL", ""))
    llm_model: str = "gpt-6.1-sol"
    # gpt-6.1-sol only accepts temperature=1 (reasoning model); other values give a 400.
    llm_temperature: float = 1.0

    # --- Upload limits (spec task 7: "Limit file size/type") ---
    max_file_size_mb: int = 10
    allowed_extensions: tuple = (".pdf",)

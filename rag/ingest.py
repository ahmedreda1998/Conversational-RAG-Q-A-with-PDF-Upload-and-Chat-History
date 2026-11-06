"""
Pipeline:
    PDF file --PyPDFLoader--> list[Document] (one per page, with metadata)
             --RecursiveCharacterTextSplitter--> smaller Documents (chunks)
             --OllamaEmbeddings--> vectors
             --Chroma--> persisted on disk

Why chunk? LLMs have limited context and retrieval works better on focused
passages than on whole 50-page documents.

"""
import os
from typing import List
import tempfile
import re
from config import Config
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

class VectorStoreClient:
    """
    Thin adapter around Chroma (spec: "Add an adapter for different vector
    stores behind a VectorStoreClient interface").
    """

    def __init__(self, config: Config, api_key: str):

        # Embeddings run locally through Ollama, so no API key is needed here
        # (api_key is still accepted because the chat LLM uses it elsewhere).
        self.embedding_model = OllamaEmbeddings(
            model=config.embedding_model, base_url=config.ollama_base_url
        )
        self.store = Chroma(
            collection_name=config.collection_name,
            embedding_function=self.embedding_model,
            persist_directory=config.persist_directory,
        )

    def add_documents(self, docs: list) -> int:
        
        docs_added = self.store.add_documents(docs)
        return len(docs_added)

    def as_retriever(self, k: int):
        """
        return self.store.as_retriever(search_kwargs={"k": k}).
        BONUS: try search_type="mmr" for more diverse results.
        """
        return self.store.as_retriever(
            search_type="mmr", search_kwargs={"k": k, "fetch_k": max(k * 4, 20)}
        )

    def is_empty(self) -> bool:
        """
        return True if the collection has no documents.
        Spec testing item: "empty store behavior". The UI should tell the
        user to upload a PDF instead of crashing. Hint: the underlying
        chroma collection has a .count() method.
        """
        return self.store._collection.count() == 0


class PdfIngestor:
    """Turns uploaded PDF files into chunks stored in the vector store."""

    def __init__(self, config: Config, vector_store: VectorStoreClient):
        """
          - keep config and vector_store
          - build RecursiveCharacterTextSplitter(chunk_size=..., chunk_overlap=...)
        """
        self.config = config
        self.vector_store = vector_store
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap
        )

    def validate_file(self, filename: str, size_bytes: int) -> None:

        if not filename.lower().endswith(tuple(self.config.allowed_extensions)):
            raise ValueError(f"File type not allowed. Please upload a file with one of the following extensions: {', '.join(self.config.allowed_extensions)}")

        if size_bytes > self.config.max_file_size_mb * 1024 * 1024:
            raise ValueError(f"File size exceeds the limit of {self.config.max_file_size_mb} MB.")

    def load_pdf(self, file_path: str) -> list:
        """
        Use PyPDFLoader(file_path).load() -> list of page Documents.

        Streamlit gives you an in-memory UploadedFile, but
        PyPDFLoader wants a path. Save the bytes to a temporary file
        (tempfile.NamedTemporaryFile) in ingest_uploaded_files, pass the path
        here, and delete the temp file afterwards (the PDF is not the API key,
        but don't leave user documents lying around either).
        """
        
        loader = PyPDFLoader(file_path)
        return loader.load()

    def clean_text(self, docs: list) -> list:
        """
        Spec task 7: sanitize text.

        for each doc, strip null bytes / control characters and
        collapse excessive whitespace in doc.page_content. Drop pages that
        become empty (scanned PDFs often have no extractable text).
        """
        cleaned_docs = []
        for doc in docs:
            # Strip null bytes and control characters
            doc.page_content = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', doc.page_content)
            # Collapse excessive whitespace
            doc.page_content = re.sub(r'\s+', ' ', doc.page_content).strip()
            # Drop pages that become empty
            if doc.page_content:
                cleaned_docs.append(doc)
        return cleaned_docs

    def split(self, docs: list) -> list:
        """Return self.splitter.split_documents(docs)."""
        return self.splitter.split_documents(docs)

    def ingest_uploaded_files(self, uploaded_files: List) -> dict:
        """
        Main entry point called by app.py.

        for each Streamlit UploadedFile:
          1. validate_file(file.name, file.size)
          2. write file.getbuffer() to a temp file, load_pdf, delete temp file
          3. clean_text -> split
          4. (BONUS citations) make sure each chunk's metadata keeps the
             source filename and page number -- PyPDFLoader already stores
             "page"; add "source" = file.name.
          5. vector_store.add_documents(chunks)
        Wrap each file in try/except so ONE bad PDF doesn't kill the batch
        (spec: "handle ingestion failures").

        Return a summary the UI can show, e.g.
            {"ingested": {"a.pdf": 42}, "failed": {"b.pdf": "reason"}}
        """

        summary = {"ingested": {}, "failed": {}}
        for file in uploaded_files:
            tmp_path = None
            try:
                self.validate_file(file.name, file.size)
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tmp_file:
                    tmp_file.write(file.getbuffer())
                    tmp_path = tmp_file.name
                docs = self.load_pdf(tmp_path)  # file is closed (and flushed) here
                docs = self.clean_text(docs)
                chunks = self.split(docs)
                for chunk in chunks:
                    chunk.metadata["source"] = file.name
                self.vector_store.add_documents(chunks)
                summary["ingested"][file.name] = len(chunks)
            except Exception as e:
                summary["failed"][file.name] = str(e)
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        return summary
    

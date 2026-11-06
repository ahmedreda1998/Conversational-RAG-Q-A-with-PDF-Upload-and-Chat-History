# Conversational RAG Q&A (Streamlit + LangChain + Chroma)

## What it does
Upload one or more PDFs, then chat with them. Each question is answered from the
uploaded documents only. Follow-up questions such as "what about its price?" are
rewritten into standalone questions using the chat history before retrieval.
Every Session ID keeps its own separate conversation.

## Architecture
The UI only talks to a `Chatbot` facade. The facade delegates to a `ChatEngine`,
so the engine can be swapped without touching the UI (Strategy pattern).

```
Streamlit UI (app.py)
      |
   Chatbot                      chatbot.py
      |
 ChatEngine (Protocol)          engines/base.py
   |-- SimpleFAQEngine          engines/simple_faq.py   (legacy keyword bot)
   |-- RagEngine                engines/rag_engine.py   (answer, answer_stream, get_sources)
   `-- DirectLlmEngine          engines/direct_llm_engine.py   (plain chat, no PDFs)
          |
     rag/pipeline.py
       - LLMClient              builds the OpenAI chat model
       - SessionHistoryStore    per-session chat history
       - RagPipelineBuilder     history-aware retriever -> QA chain
          |
     rag/ingest.py
       - PdfIngestor            validate -> load -> clean -> split -> store
       - VectorStoreClient      Chroma + Ollama embeddings
```

Per question:
1. If there is chat history, the LLM rewrites the follow-up as a standalone question. With no history (first question) it goes to retrieval unchanged.
2. The standalone question retrieves the top-k chunks from Chroma (MMR search).
3. The LLM answers from those chunks, with the chat history in the prompt.
4. If the vector store is empty, `RagEngine` replies "please upload a PDF" without calling the LLM.

`app.py` is the composition root: it is the only place that chooses concrete classes.

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Embeddings run locally through [Ollama](https://ollama.com), so Ollama must be
running and the embedding model pulled once:
```bash
ollama pull hf.co/mixedbread-ai/mxbai-embed-large-v1:F16
```

## Environment / API key
Enter your key in the **API Key** field in the sidebar (password input). It is
held in `st.session_state` only and never written to disk. Changing it rebuilds
the chatbots; chat histories are kept.

As a development convenience the app falls back to the project's `.env`
(`python-dotenv`) when the field is empty:
```
OPENAI_API_KEY=...
OPENAI_BASE_URL=...
```
`OPENAI_BASE_URL` is always read from `.env`. The key is passed explicitly to the
client and is not stored in `Config`. Keep `.env` out of version control.
Error messages shown in the UI contain only a friendly text or the exception
type, never the raw exception message, because that can echo the key.

## Run
```bash
streamlit run app.py
```
1. Enter your API key in the sidebar (or set `OPENAI_API_KEY` in `.env`); make sure `.env` has `OPENAI_BASE_URL`.
2. Pick a **Mode** in the sidebar:
   - **Chat with my PDFs (RAG):** upload PDFs, click **Process PDFs**, then ask questions about them. Answers show their sources.
   - **Ask the model directly:** no PDFs; the model answers from its own knowledge.
3. Use different Session IDs for separate conversations. The two modes keep separate chat histories, so switching mode switches to that mode's conversation.

## Configuration
All settings are in `config.py`:

| Setting | Default | Meaning |
|---|---|---|
| `persist_directory` | `.chroma/student-rag` | where Chroma stores vectors on disk |
| `collection_name` | `student-rag` | Chroma collection |
| `chunk_size` / `chunk_overlap` | 1000 / 150 | text splitting |
| `embedding_model` | `hf.co/mixedbread-ai/mxbai-embed-large-v1:F16` | Ollama embedding model |
| `ollama_base_url` | `http://localhost:11434` | Ollama server |
| `retriever_k` | 4 | chunks retrieved per question |
| `openai_base_url` | from `.env` | chat endpoint (`OPENAI_BASE_URL`) |
| `llm_model` | `gpt-6.1-sol` | chat model; confirmed available on the project endpoint |
| `llm_temperature` | 1.0 | `gpt-6.1-sol` rejects any other value (400 error) |
| `max_file_size_mb` | 10 | upload size limit |
| `allowed_extensions` | `.pdf` | upload type limit |

Uploads are validated for type and size, control characters are stripped, pages
with no extractable text (for example scanned PDFs) are dropped, and one bad
PDF does not stop the rest of the batch.


# PDF Q&A App (Groq + Google Gemini Embeddings)

A Streamlit app that answers questions about your own PDF documents using
**RAG (Retrieval-Augmented Generation)**: your PDFs are chunked, embedded
with Google's `gemini-embedding-001`, indexed in a local FAISS vector store,
and retrieved chunks are passed as context to Groq's `openai/gpt-oss-20b`
model to generate the final answer.

## Screenshots

**Upload a PDF and ask a question:**

![Upload PDF](screenshots/upload.png)

**Answer generated from the document, with the retrieved source chunk shown:**

![Answer with similarity search](screenshots/answer.png)

## How it works

1. Upload one or more PDFs through the file uploader (saved to `us_cencus/`).
2. Click **"Creating Vector Store"** — the PDFs are loaded, split into
   ~1000-character chunks, embedded, and indexed in FAISS.
3. Type a question in **"Ask the document"** — the top matching chunks are
   retrieved and passed to the LLM, which answers using only that context.
4. Expand **"Document Similarity Search"** to see exactly which chunks the
   answer was based on.

## Setup

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file in this folder:

```env
GROQ_API_KEY=your_groq_api_key
GOOGLE_API_KEY=your_google_api_key
```

Run it:

```bash
streamlit run app.py
```

## Stack

| Piece | Tool |
|---|---|
| UI | Streamlit |
| Orchestration | LangChain (`langchain-classic` for the retrieval chain helpers) |
| Chat model | Groq — `openai/gpt-oss-20b` |
| Embeddings | Google Generative AI — `gemini-embedding-001` |
| Vector store | FAISS (in-memory) |
| PDF parsing | pypdf |

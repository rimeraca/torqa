# 📚 RAG Assistant

A local, offline document Q&A assistant built with Microsoft Foundry Local and the RAG (Retrieval-Augmented Generation) pattern.

Built as part of the **Microsoft Türkiye Summer Program 2026**.

---

## What it does

Upload any PDF or TXT document and ask questions about it in natural language. The assistant finds the most relevant parts of your document and generates accurate, grounded answers using a local AI model — no internet required.

---

## How it works

1. **Upload** — User uploads a PDF or TXT file
2. **Chunk** — Document is split into lines/passages
3. **Embed** — Each chunk is converted into a vector using a sentence embedding model
4. **Search** — User's question is also embedded and compared against all chunks using cosine similarity
5. **Generate** — The top matching chunks are sent to a local LLM (via Microsoft Foundry Local) which generates a natural language answer

---

## Tech Stack

- **Microsoft Foundry Local** — on-device LLM inference (phi-3.5-mini), no cloud needed
- **sentence-transformers** — text embedding model (all-MiniLM-L6-v2)
- **SQLite** — lightweight local database for storing document chunks and embeddings
- **Streamlit** — web interface
- **Python** — core language

---

## Project Structure

rag-assistant/
├── app.py          # Main Streamlit web app
├── ingest.py       # Loads documents and saves embeddings to database
├── database.py     # Creates the SQLite database schema
├── knowledge.db    # Local SQLite database
├── documents/      # Folder for knowledge base text files
│   ├── sample.txt
│   ├── clubs.txt
│   └── admissions.txt
└── README.md

---

## Setup & Installation

### 1. Install Microsoft Foundry Local
Follow the
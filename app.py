import streamlit as st
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import numpy as np
from pypdf import PdfReader

# Page config
st.set_page_config(
    page_title="Torqa",
    page_icon="logo.png",
    layout="centered",
)

# Sidebar
with st.sidebar:
    st.image("logo.png", width=200)
    st.write("A local offline RAG assistant built with Microsoft Foundry Local.")
    st.divider()
    st.write("**How it works:**")
    st.write("1. Upload a PDF or TXT file")
    st.write("2. Ask any question about it")
    st.write("3. Get an AI-generated answer")
    st.divider()
    st.write("**Tech Stack:**")
    st.write("- Microsoft Foundry Local")
    st.write("- sentence-transformers")
    st.write("- SQLite")
    st.write("- Streamlit")
    st.divider()
    st.write("Built for **Microsoft Türkiye Summer Program 2026**")

# Main page
st.image("logo.png", width=200)
st.title("Torqa")

# Connect to Foundry Local
client = OpenAI(
    base_url="http://127.0.0.1:63429/v1",
    api_key="local"
)
model = "phi-3.5-mini-instruct-trtrtx-gpu:2"

# Load embedding model
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()

# File upload
uploaded_files = st.file_uploader("📄 Upload PDF or TXT files", type=["pdf", "txt"], accept_multiple_files=True)

if uploaded_files:
    chunks = []
    sources = []

    for uploaded_file in uploaded_files:
        if uploaded_file.name.endswith(".pdf"):
            reader = PdfReader(uploaded_file)
            document = ""
            for page in reader.pages:
                document += page.extract_text()
        else:
            document = uploaded_file.read().decode("utf-8")

        file_chunks = document.split("\n")
        file_chunks = [chunk.strip() for chunk in file_chunks if chunk.strip()]
        chunks.extend(file_chunks)
        sources.extend([uploaded_file.name] * len(file_chunks))

    st.success(f"✅ Loaded {len(chunks)} chunks from {len(uploaded_files)} file(s)")

    # Generate embeddings
    chunk_embeddings = embedder.encode(chunks)

    # Chat input
    question = st.chat_input("Ask a question about your documents...")

    if question:
        with st.chat_message("user"):
            st.write(question)

        # Find top 2 chunks
        question_embedding = embedder.encode(question)
        similarities = np.dot(chunk_embeddings, question_embedding.flatten())
        top_indices = np.argsort(similarities)[-2:][::-1]
        top_chunks = [chunks[i] for i in top_indices]
        top_sources = [sources[i] for i in top_indices]
        context = "\n".join(top_chunks)

        # Calculate confidence score
        top_score = float(similarities[top_indices[0]])
        max_possible = float(np.max(similarities))
        confidence = int((top_score / max_possible) * 100) if max_possible > 0 else 0

        # Get AI answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "Answer using ONLY this information:\n" + context},
                        {"role": "user", "content": question}
                    ]
                )
            st.write(response.choices[0].message.content)
            st.write("---")

            # Show confidence
            if confidence >= 75:
                st.success(f"🟢 High Confidence ({confidence}%)")
            elif confidence >= 50:
                st.warning(f"🟡 Medium Confidence ({confidence}%)")
            else:
                st.error(f"🔴 Low Confidence ({confidence}%)")

            st.write("📄 **Sources:** " + ", ".join(set(top_sources)))

            # Expandable retrieved chunks
            with st.expander("🔍 View Retrieved Context"):
                for i, chunk in enumerate(top_chunks):
                    st.write(f"**Chunk {i+1}:**")
                    st.info(chunk)
import streamlit as st
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import numpy as np
from pypdf import PdfReader

# Page setup
st.title("📚 Codex")
st.write("Upload a PDF or TXT file and ask questions.")

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
uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])

if uploaded_file is not None:
    # Extract text depending on file type
    if uploaded_file.name.endswith(".pdf"):
        reader = PdfReader(uploaded_file)
        document = ""
        for page in reader.pages:
            document += page.extract_text()
    else:
        document = uploaded_file.read().decode("utf-8")

    # Split into chunks
    chunks = document.split("\n")
    chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
    sources = [uploaded_file.name] * len(chunks)

    st.success(f"Loaded {len(chunks)} chunks from {uploaded_file.name}")

    # Generate embeddings
    chunk_embeddings = embedder.encode(chunks)

    # User input at bottom
    question = st.chat_input("Ask a question about the document...")

    if question:
        # Show user message
        with st.chat_message("user"):
            st.write(question)

        # Find top 2 chunks
        question_embedding = embedder.encode(question)
        similarities = np.dot(chunk_embeddings, question_embedding.flatten())
        top_indices = np.argsort(similarities)[-2:][::-1]
        top_chunks = [chunks[i] for i in top_indices]
        top_sources = [sources[i] for i in top_indices]
        context = "\n".join(top_chunks)

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
            st.write("📄 **Sources:**")
            for source in set(top_sources):
                st.write(f"- {source}")
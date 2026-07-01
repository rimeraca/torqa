import streamlit as st
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import sqlite3
import numpy as np
from pypdf import PdfReader

#page setup
st.title("📚 RAG Assistant")
st.write("Upload a PDF or TXT file and ask questions.")

#connect to Foundry Local
client = OpenAI(
    base_url="http://127.0.0.1:63429/v1",
    api_key="local"
)
model = "phi-3.5-mini-instruct-trtrtx-gpu:2"

#load embedding model
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()

#file upload
uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])

if uploaded_file is not None:
    # extract text depending on file type
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

    # Generate embeddings for chunks (in memory, not saved to DB)
    chunk_embeddings = embedder.encode(chunks)

    # User input
    question = st.text_input("Your question:")

    if question:
        question_embedding = embedder.encode(question)
        
        # Find top 2 chunks
        similarities = np.dot(chunk_embeddings, question_embedding.T)
        top_indices = np.argsort(similarities)[-2:][::-1]
        top_chunks = [chunks[i] for i in top_indices]
        top_sources = [sources[i] for i in top_indices]
        context = "\n".join(top_chunks)
        
        # Get AI answer
        with st.spinner("Thinking..."):
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": f"Answer using ONLY this information:\n{context}"},
                    {"role": "user", "content": question}
                ]
            )
        
        st.write("### Answer:")
        st.write(response.choices[0].message.content)
        st.write("---")
        st.write("📄 **Sources:**")
        for source in set(top_sources):
            st.write(f"- {source}")
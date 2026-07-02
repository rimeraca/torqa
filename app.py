import streamlit as st
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import numpy as np
from pypdf import PdfReader
from fpdf import FPDF

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
    st.write("1. Upload one or more PDF or TXT file")
    st.write("2. Ask any question about them")
    st.write("3. Get a precise AI-generated answer")
    st.write("4. See confidence score and source citations")
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
col1, col2 = st.columns([4, 1])
with col1:
    st.write("Upload documents and ask questions - fully offline.")
with col2:
    if st.button("🔄 Clear"):
        st.rerun()

# Connect to Foundry Local
client = OpenAI(
    base_url="http://127.0.0.1:63429/v1",
    api_key="local"
)
model = "phi-3.5-mini-instruct-trtrtx-gpu:2"

#check if foundary local is running
try:
    import urllib.request
    urllib.request.urlopen("http://127.0.0.1:63429/openai/status", timeout=2)
except Exception:
    st.error("❌ Foundry Local is not running. Please start it with: `foundry model run phi-3.5-mini`")
    st.stop()

# Load embedding model
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()

def generate_pdf_report(history, summary):
    pdf = FPDF()
    pdf.set_margins(10, 10, 10)
    pdf.add_page()

    def clean(text):
        if not text:
            return ""
        result = ""
        for char in str(text):
            try:
                char.encode("latin-1")
                result += char
            except UnicodeEncodeError:
                result += "?"
        return result

    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(190, 10, "Torqa - Session Report", ln=True, align="C")
    pdf.ln(5)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(190, 10, "Document Summary:", ln=True)
    pdf.set_font("Helvetica", size=10)
    pdf.multi_cell(190, 8, clean(summary))
    pdf.ln(5)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(190, 10, "Q&A History:", ln=True)

    for i, item in enumerate(history):
        pdf.set_font("Helvetica", "B", 10)
        pdf.multi_cell(190, 8, clean(f"Q{i+1}: {item['question']}"))
        pdf.set_font("Helvetica", size=10)
        raw_answer = str(item.get('answer', 'N/A'))
        safe_answer = raw_answer.encode('ascii', errors='replace').decode('ascii')
        pdf.set_x(10)
        pdf.multi_cell(170, 8, "Answer: " + safe_answer)
        pdf.multi_cell(190, 8, clean(f"Confidence: {item['confidence']}%"))
        pdf.multi_cell(190, 8, clean(f"Sources: {', '.join(item['sources'])}"))
        pdf.ln(3)

    return pdf.output()

#initialize answer history
if "history" not in st.session_state:
    st.session_state.history = []

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

    #generate document summary
    file_key = "-".join([f.name for f in uploaded_files])
    if "summary" not in st.session_state or st.session_state.get("summary_key") != file_key:
        with st.spinner("Summarizing document..."):
            summary_context = " ".join(chunks[:10] + chunks[len(chunks)//2:len(chunks)//2 + 10])
            summary_response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Summarize the following document in 2 sentences max. Be concise."},
                    {"role": "user", "content": summary_context}
                ]
            )
            st.session_state.summary = summary_response.choices[0].message.content
            st.session_state.summary_key = file_key

    st.info("📋 **Document Summary:** " + st.session_state.summary)
        

    # Generate embeddings
    chunk_embeddings = embedder.encode(chunks)

    # Display previous answers from history
    for item in st.session_state.history:
        with st.chat_message("user"):
            st.write(item["question"])
        with st.chat_message("assistant"):
            st.write(item["answer"])
            st.write("---")
            if item["confidence"] >= 75:
                st.success(f"🟢 High Confidence ({item['confidence']}%)")
            elif item["confidence"] >= 50:
                st.warning(f"🟡 Medium Confidence ({item['confidence']}%)")
            else:
                st.error(f"🔴 Low Confidence ({item['confidence']}%)")
            st.write("📄 **Sources:** " + ", ".join(item["sources"]))

    # Chat input
    question = st.chat_input("Ask a question about your documents...", key="man_input")

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
                        {"role": "system", "content": "You are a precise assistant. Answer the question using ONLY the information provided below. Be direct and concise. Do not add explanations, assumptions, or information not present in the context. If the answer is not in the context, say 'I don't have that information.'\n\nContext:\n" + context},
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

            #save to history
            st.session_state.history.append({
                "question": question,
                "answer": response.choices[0].message.content,
                "confidence": confidence,
                "sources": list(set(top_sources)),
            })
#show history
if st.session_state.history:
    st.divider()
    st.write("### 📜 Session History")
    # Export button
    pdf_data = generate_pdf_report(st.session_state.history, st.session_state.get("summary", ""))
    st.download_button(
        label="📥 Download Session Report",
        data=bytes(pdf_data),
        file_name="torqa_report.pdf",
        mime="application/pdf"
    )
    for i, item in enumerate(reversed(st.session_state.history)):
        with st.expander(f"Q{len(st.session_state.history) - i}: {item['question']}"):
            st.write(f"**Answer:** {item['answer']}")
            st.write("**Sources:**", ", ".join(item["sources"]))
            if item['confidence'] >= 75:
                st.success(f"🟢 High Confidence ({item['confidence']}%)")
            elif item['confidence'] >= 50:
                st.warning(f"🟡 Medium Confidence ({item['confidence']}%)")
            else:
                st.error(f"🔴 Low Confidence ({item['confidence']}%)")
import streamlit as st
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import numpy as np
from pypdf import PdfReader
from fpdf import FPDF
from datetime import datetime
from rank_bm25 import BM25Okapi
import urllib.request

# Initialize session state
if "chats" not in st.session_state:
    st.session_state.chats = {"Chat 1": []}
    st.session_state.current_chat = "Chat 1"
if "history" not in st.session_state:
    st.session_state.history = st.session_state.chats[st.session_state.current_chat]
if "remaining" not in st.session_state:
    st.session_state.remaining = None

# Page config
st.set_page_config(
    page_title="Torqa",
    page_icon="logo.png",
    layout="centered",
)

st.markdown("""
<style>
    .stApp { background-color: #0a1628; }
    [data-testid="stSidebar"] { background-color: #071020; }
    .stMarkdown, p, h1, h2, h3, label { color: #e8edf5 !important; }
    h1 { color: #4a9eff !important; font-family: 'Segoe UI', sans-serif !important; }
    [data-testid="stChatMessage"] { background-color: #0f2040 !important; border: 1px solid #1e3a5f !important; border-radius: 8px !important; }
    .stButton > button { background-color: #1a3a6b !important; color: #4a9eff !important; border: 1px solid #2a5298 !important; border-radius: 6px !important; }
    .stButton > button:hover { background-color: #2a5298 !important; color: #ffffff !important; }
    hr { border-color: #1e3a5f !important; }
    section[data-testid="stBottom"] > div { background-color: #0a1628 !important; }
    [data-testid="stChatInput"] { background-color: #0f2040 !important; border: 1px solid #2a5298 !important; }
    textarea[data-testid="stChatInputTextArea"] { background-color: #0f2040 !important; color: #e8edf5 !important; }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("logo.png", width=200)
    st.write("A local offline RAG assistant built with Microsoft Foundry Local.")
    st.divider()
    st.write("**How it works:**")
    st.write("1. Upload one or more PDF or TXT files")
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
    st.divider()
    st.write("**💬 Chats**")
    for chat_name in list(st.session_state.chats.keys()):
        col_a, col_b, col_c = st.columns([3, 1, 1])
        with col_a:
            if st.button(chat_name, key=f"chat_{chat_name}", use_container_width=True):
                st.session_state.current_chat = chat_name
                st.session_state.history = st.session_state.chats[chat_name]
                st.session_state.remaining = None
                st.rerun()
        with col_b:
            if st.button("✏️", key=f"rename_{chat_name}"):
                st.session_state.remaining = chat_name
        with col_c:
            if st.button("🗑️", key=f"del_{chat_name}"):
                if len(st.session_state.chats) > 1:
                    del st.session_state.chats[chat_name]
                    st.session_state.current_chat = list(st.session_state.chats.keys())[0]
                    st.session_state.history = st.session_state.chats[st.session_state.current_chat]
                    st.rerun()

    if st.button("➕ New Chat", use_container_width=True):
        chat_count = len(st.session_state.chats) + 1
        new_chat_name = f"Chat {chat_count}"
        st.session_state.chats[new_chat_name] = []
        st.session_state.current_chat = new_chat_name
        st.session_state.history = st.session_state.chats[new_chat_name]
        st.rerun()

    if st.session_state.remaining:
        new_name = st.text_input("New Name:", value=st.session_state.remaining, key="rename_input")
        if st.button("✅ Save", use_container_width=True):
            if new_name and new_name != st.session_state.remaining:
                chats = st.session_state.chats
                chats[new_name] = chats.pop(st.session_state.remaining)
                st.session_state.current_chat = new_name
                st.session_state.history = chats[new_name]
                st.session_state.remaining = None
                st.rerun()
            else:
                st.warning("Please enter a valid new name.")

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
foundry_endpoint = None
known_ports = [63429, 58817, 54170, 53035, 61573, 63086, 57626, 57861, 5273, 8080]
for port in known_ports:
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/openai/status", timeout=1)
        foundry_endpoint = f"http://127.0.0.1:{port}/v1"
        break
    except:
        continue

if not foundry_endpoint:
    for port in range(50000, 65000, 100):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/openai/status", timeout=0.3)
            foundry_endpoint = f"http://127.0.0.1:{port}/v1"
            break
        except:
            continue

if not foundry_endpoint:
    st.error("❌ Foundry Local is not running. Please start it with: `foundry model run phi-4-mini`")
    st.stop()

model = "Phi-4-mini-instruct-cuda-gpu:5"
client = OpenAI(base_url=foundry_endpoint, api_key="local")

@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()

def build_prompt(context, question):
    transcript_keywords = ["gpa", "cgpa", "ects", "credits", "grade point", "letter grade", "transcript"]
    is_transcript = any(kw in context.lower() for kw in transcript_keywords)
    
    if is_transcript:
        return [{"role": "user", "content": f"Here is a university transcript:\n\n{context}\n\nIn university transcripts, the format is: COURSE_CODE COURSE_NAME CREDITS LETTER_GRADE GRADE_POINTS ECTS\n\nAnswer this question: {question}"}]
    else:
        return [
            {"role": "system", "content": "Answer ONLY using the exact text below. Extract the answer directly from the text.\n\nText:\n" + context},
            {"role": "user", "content": question}
        ]


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

# File upload
with st.expander("📎 Attach files"):
    uploaded_files = st.file_uploader("", type=["pdf", "txt"], accept_multiple_files=True, label_visibility="collapsed")

# General chat mode
if not uploaded_files:
    general_question = st.chat_input("Ask anything or upload a document...", key="general_input")
    if general_question:
        with st.chat_message("user"):
            st.write(general_question)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant. Answer questions accurately and concisely."},
                        {"role": "user", "content": general_question}
                    ]
                )
            st.write(response.choices[0].message.content)

# Document mode
if uploaded_files:
    chunks = []
    sources = []

    for uploaded_file in uploaded_files:
        try:
            if uploaded_file.name.endswith(".pdf"):
                reader = PdfReader(uploaded_file)
                document = ""
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        document += text
            else:
                document = uploaded_file.read().decode("utf-8")

            if not document.strip():
                st.warning(f"⚠️ Could not extract text from **{uploaded_file.name}**.")
                continue

            lines = [line.strip() for line in document.split("\n") if line.strip()]
            avg_line_length = sum(len(line.split()) for line in lines) / len(lines) if lines else 0

            if avg_line_length < 10:
                file_chunks = lines
            else:
                file_chunks = []
                current_chunk = []
                current_length = 0
                for line in lines:
                    current_chunk.append(line)
                    current_length += len(line.split())
                    if current_length >= 50 or line.endswith(".") and current_length >= 20:
                        file_chunks.append(" ".join(current_chunk))
                        current_chunk = []
                        current_length = 0
                if current_chunk:
                    file_chunks.append(" ".join(current_chunk))

            chunks.extend(file_chunks)
            sources.extend([uploaded_file.name] * len(file_chunks))

        except Exception as e:
            st.error(f"⚠️ Could not read **{uploaded_file.name}**: {str(e)}")
            continue

    total_words = sum(len(chunk.split()) for chunk in chunks)
    estimated_pages = round(total_words / 250)

    if len(chunks) > 500:
        st.warning(f"⚠️ Large document — {total_words:,} words (~{estimated_pages} pages). Processing may take longer.")
    else:
        st.success(f"✅ {len(uploaded_files)} file(s) loaded — {total_words:,} words (~{estimated_pages} pages)")

    file_key = "-".join([f.name for f in uploaded_files])
    if "summary" not in st.session_state or st.session_state.get("summary_key") != file_key:
        with st.spinner("Summarizing document..."):
            summary_context = " ".join(chunks[:10] + chunks[len(chunks)//2:len(chunks)//2 + 10])
            summary_response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Summarize the following document in 2 sentences. No notes, no disclaimers."},
                    {"role": "user", "content": summary_context}
                ]
            )
            st.session_state.summary = summary_response.choices[0].message.content
            st.session_state.summary_key = file_key

    st.info("📋 **Document Summary:** " + st.session_state.summary)

    chunk_embeddings = embedder.encode(chunks)

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

    question = st.chat_input("Ask a question about your documents...", key="man_input")

    if question:
        with st.chat_message("user"):
            st.write(question)

        question_embedding = embedder.encode(question)
        dense_scores = np.dot(chunk_embeddings, question_embedding.flatten())
        dense_scores_norm = (dense_scores - dense_scores.min()) / (dense_scores.max() - dense_scores.min() + 1e-9)

        tokenized_chunks = [chunk.lower().split() for chunk in chunks]
        bm25 = BM25Okapi(tokenized_chunks)
        bm25_scores = bm25.get_scores(question.lower().split())
        bm25_scores_norm = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-9)

        hybrid_scores = 0.6 * dense_scores_norm + 0.4 * bm25_scores_norm
        top_indices = np.argsort(hybrid_scores)[-2:][::-1]
        top_chunks = [chunks[i] for i in top_indices]
        top_sources = [sources[i] for i in top_indices]
        context = "\n".join(top_chunks)

        top_score = float(hybrid_scores[top_indices[0]])
        max_possible = float(np.max(hybrid_scores))
        confidence = int((top_score / max_possible) * 100) if max_possible > 0 else 0

        with st.chat_message("assistant"):
            answer_placeholder = st.empty()
            full_answer = ""
            stream = client.chat.completions.create(
                model=model,
                messages=build_prompt(context, question),
                stream=True
            )
            try:
                for chunk_part in stream:
                    if chunk_part.choices[0].delta.content is not None:
                        full_answer += chunk_part.choices[0].delta.content
                        answer_placeholder.markdown(full_answer + "▌")
            except Exception:
                pass
            answer_placeholder.markdown(full_answer)

            st.write("---")

            if confidence >= 75:
                st.success(f"🟢 High Confidence ({confidence}%)")
            elif confidence >= 50:
                st.warning(f"🟡 Medium Confidence ({confidence}%)")
            else:
                st.error(f"🔴 Low Confidence ({confidence}%)")

            st.write("📄 **Sources:** " + ", ".join(set(top_sources)))

            with st.expander("🔍 View Retrieved Context"):
                for i, chunk in enumerate(top_chunks):
                    st.write(f"**Chunk {i+1}:**")
                    st.info(chunk)

            st.session_state.history.append({
                "question": question,
                "answer": full_answer,
                "confidence": confidence,
                "sources": list(set(top_sources)),
                "timestamp": datetime.now()
            })
            st.session_state.chats[st.session_state.current_chat] = st.session_state.history

if st.session_state.history:
    st.divider()
    st.write("### 📜 Session History")
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
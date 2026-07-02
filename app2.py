import streamlit as st
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import numpy as np
from pypdf import PdfReader
from datetime import datetime

# Page config
st.set_page_config(
    page_title="Torqa",
    page_icon="logo.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Full app background */
    .stApp {
        background-color: #f8f7f4;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e5e5e5;
        width: 260px !important;
    }
    
    /* Sidebar text */
    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] label {
        color: #1a1a1a !important;
        font-size: 14px !important;
    }
    
    /* Sidebar buttons */
    [data-testid="stSidebar"] .stButton > button {
        background-color: transparent !important;
        color: #1a1a1a !important;
        border: none !important;
        text-align: left !important;
        padding: 8px 12px !important;
        border-radius: 8px !important;
        width: 100% !important;
        font-size: 14px !important;
    }
    
    [data-testid="stSidebar"] .stButton > button:hover {
        background-color: #f0f0f0 !important;
    }
    
    /* Main area */
    .main .block-container {
        padding: 0 !important;
        max-width: 100% !important;
    }
    
    /* Chat messages */
    [data-testid="stChatMessage"] {
        background-color: transparent !important;
        border: none !important;
        padding: 8px 0 !important;
    }
    
    /* Chat input */
    [data-testid="stChatInput"] {
        background-color: #ffffff !important;
        border: 1px solid #e5e5e5 !important;
        border-radius: 12px !important;
    }
    
    section[data-testid="stBottom"] > div {
        background-color: #f8f7f4 !important;
        padding: 16px !important;
    }
    
    /* Hide streamlit branding */
    .viewerBadge_container__r5tak {display: none;}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "chats" not in st.session_state:
    st.session_state.chats = {"New Chat": []}
    st.session_state.current_chat = "New Chat"

if "history" not in st.session_state:
    st.session_state.history = st.session_state.chats[st.session_state.current_chat]

if "remaining" not in st.session_state:
    st.session_state.remaining = None

# Connect to Foundry Local
client = OpenAI(
    base_url="http://127.0.0.1:63429/v1",
    api_key="local"
)
model = "phi-3.5-mini-instruct-trtrtx-gpu:2"

# Check Foundry Local
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

# SIDEBAR
with st.sidebar:
    # Logo and title
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image("logo.png", width=35)
    with col2:
        st.markdown("### Torqa")
    
    st.divider()
    
    # New chat button
    if st.button("✦ New Chat", use_container_width=True, key="new_chat_btn"):
        chat_count = len(st.session_state.chats) + 1
        new_chat_name = f"Chat {chat_count}"
        st.session_state.chats[new_chat_name] = []
        st.session_state.current_chat = new_chat_name
        st.session_state.history = st.session_state.chats[new_chat_name]
        st.rerun()

    st.markdown("**Chats**")
    
    # Chat list
    for chat_name in list(st.session_state.chats.keys()):
        is_active = chat_name == st.session_state.current_chat
        col_a, col_b, col_c = st.columns([5, 1, 1])
        with col_a:
            label = f"**{chat_name}**" if is_active else chat_name
            if st.button(label, key=f"chat_{chat_name}", use_container_width=True):
                st.session_state.current_chat = chat_name
                st.session_state.history = st.session_state.chats[chat_name]
                st.session_state.remaining = None
                st.rerun()
        with col_b:
            if st.button("✏️", key=f"ren_{chat_name}"):
                st.session_state.remaining = chat_name
        with col_c:
            if st.button("🗑", key=f"del_{chat_name}"):
                if len(st.session_state.chats) > 1:
                    del st.session_state.chats[chat_name]
                    st.session_state.current_chat = list(st.session_state.chats.keys())[0]
                    st.session_state.history = st.session_state.chats[st.session_state.current_chat]
                    st.rerun()

    # Rename
    if st.session_state.remaining:
        new_name = st.text_input("Rename:", value=st.session_state.remaining, key="rename_input")
        if st.button("✅ Save", use_container_width=True):
            if new_name and new_name != st.session_state.remaining:
                chats = st.session_state.chats
                chats[new_name] = chats.pop(st.session_state.remaining)
                st.session_state.current_chat = new_name
                st.session_state.history = chats[new_name]
                st.session_state.remaining = None
                st.rerun()

    st.divider()
    st.caption("Built for Microsoft Türkiye 2026")

# MAIN AREA
# File upload at top
with st.expander("📎 Attach files"):
    uploaded_files = st.file_uploader("", type=["pdf", "txt"], accept_multiple_files=True, label_visibility="collapsed")

# General chat mode
if not uploaded_files:
    for item in st.session_state.history:
        with st.chat_message("user"):
            st.write(item["question"])
        with st.chat_message("assistant"):
            st.write(item["answer"])

    general_question = st.chat_input("Ask anything or attach a document...", key="general_input")
    if general_question:
        with st.chat_message("user"):
            st.write(general_question)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant. Answer questions accurately and concisely in 2-3 sentences maximum."},
                        {"role": "user", "content": general_question}
                    ]
                )
            answer = response.choices[0].message.content
            st.write(answer)
        
        st.session_state.history.append({
            "question": general_question,
            "answer": answer,
            "confidence": 100,
            "sources": ["General Knowledge"],
            "timestamp": datetime.now()
        })
        st.session_state.chats[st.session_state.current_chat] = st.session_state.history

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

            file_chunks = document.split("\n")
            file_chunks = [chunk.strip() for chunk in file_chunks if chunk.strip()]
            chunks.extend(file_chunks)
            sources.extend([uploaded_file.name] * len(file_chunks))

        except Exception as e:
            st.error(f"⚠️ Could not read **{uploaded_file.name}**: {str(e)}")
            continue

    total_words = sum(len(chunk.split()) for chunk in chunks)
    estimated_pages = round(total_words / 250)
    st.success(f"✅ {len(uploaded_files)} file(s) loaded — {total_words:,} words (~{estimated_pages} pages)")

    # Summary
    file_key = "-".join([f.name for f in uploaded_files])
    if "summary" not in st.session_state or st.session_state.get("summary_key") != file_key:
        with st.spinner("Summarizing..."):
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

    st.info("📋 " + st.session_state.summary)

    chunk_embeddings = embedder.encode(chunks)

    # Display history
    for item in st.session_state.history:
        with st.chat_message("user"):
            st.write(item["question"])
        with st.chat_message("assistant"):
            st.write(item["answer"])
            if item["confidence"] >= 75:
                st.success(f"🟢 {item['confidence']}% confidence")
            elif item["confidence"] >= 50:
                st.warning(f"🟡 {item['confidence']}% confidence")
            else:
                st.error(f"🔴 {item['confidence']}% confidence")
            st.caption("📄 " + ", ".join(item["sources"]))

    question = st.chat_input("Ask about your documents...", key="doc_input")

    if question:
        with st.chat_message("user"):
            st.write(question)

        question_embedding = embedder.encode(question)
        similarities = np.dot(chunk_embeddings, question_embedding.flatten())
        top_indices = np.argsort(similarities)[-2:][::-1]
        top_chunks = [chunks[i] for i in top_indices]
        top_sources = [sources[i] for i in top_indices]
        context = "\n".join(top_chunks)

        top_score = float(similarities[top_indices[0]])
        max_possible = float(np.max(similarities))
        confidence = int((top_score / max_possible) * 100) if max_possible > 0 else 0

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a precise assistant. Answer using ONLY this information. Be direct and concise. If not in context say 'I don't have that information.'\n\nContext:\n" + context},
                        {"role": "user", "content": question}
                    ]
                )
            answer = response.choices[0].message.content
            st.write(answer)
            
            if confidence >= 75:
                st.success(f"🟢 {confidence}% confidence")
            elif confidence >= 50:
                st.warning(f"🟡 {confidence}% confidence")
            else:
                st.error(f"🔴 {confidence}% confidence")
            
            st.caption("📄 " + ", ".join(set(top_sources)))
            
            with st.expander("🔍 Sources"):
                for i, chunk in enumerate(top_chunks):
                    st.caption(f"**{i+1}.** {chunk}")

        st.session_state.history.append({
            "question": question,
            "answer": answer,
            "confidence": confidence,
            "sources": list(set(top_sources)),
            "timestamp": datetime.now()
        })
        st.session_state.chats[st.session_state.current_chat] = st.session_state.history
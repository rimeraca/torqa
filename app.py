from sentence_transformers import CrossEncoder
import streamlit as st
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import numpy as np
from pypdf import PdfReader
from datetime import datetime
from rank_bm25 import BM25Okapi
import pdfplumber
import urllib.request

st.set_page_config(
    page_title="Torqa",
    page_icon="logo.png",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
        [data-testid="stSidebar"] [data-testid="stToggle"] label {
        color: #1a1a1a !important;
        font-size: 0.9rem !important;
    }
        section[data-testid="stSidebar"] {
                min-width: 220px !important;
                max-width: 240px !important;
                transform: translateX(0) !important;
                background-color: #ffffff;
                
    }
            button:has([data-testid="stIconMaterial"][translate="no"]) {
        display: none !important;
    }
            button[data-testid="baseButton-header"] {
        display: none !important;
    }
            [data-testid="stSidebarCollapseButton"] {
        display: none !important;
            
        section[data-testid="stSidebar"] {
                transform: translateX(0) !important;
                background-color: #ffffff;
    }
    }
    }
            [data-testid="stSidebar"] * {
        color: #000000 !important;
    }
            [data-testid="stSidebar"] {
        display: block !important;
        visibility: visible !important;
        transform: none !important;
        background-color: #ffffff !important;
    }
            button[kind="header"] {
        display: block !important;
        visibility: visible !important;
    }
    
    [data-testid="collapsedControl"] {
        display: block !important;
        visibility: visible !important;
    }
            [data-testid="stSidebarCollapseButton"] {
        display: block !important;
        visibility: visible !important;
    }
    [data-testid="stToolbar"] { display: none !important; }
    section[data-testid="stBottom"] { background-color: #f0f0f0 !important; }
    section[data-testid="stBottom"] > div { background-color: #f0f0f0 !important; }
    [data-testid="stChatInput"] { background-color: #ffffff !important; border: 1px solid #e0e0e0 !important; border-radius: 12px !important; }
    [data-testid="stChatInputTextArea"] { color: #1a1a1a !important; background-color: #ffffff !important; }
    [data-testid="stChatMessageAvatarUser"] { display: none !important; }
    [data-testid="stChatMessageAvatarAssistant"] { display: none !important; }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) { background-color: #ffffff !important; border-radius: 12px !important; padding: 12px 16px !important; margin: 4px 0 !important; border: 1px solid #e5e5e5 !important; }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) { background-color: #f5f5f5 !important; border-radius: 12px !important; padding: 12px 16px !important; margin: 4px 0 !important; border: none !important; }
    .block-container { padding-top: 2rem !important; padding-left: 2rem !important; padding-right: 2rem !important; }
    .main * { color: #1a1a1a !important; }
    [data-testid="stMarkdownContainer"] p { color: #1a1a1a !important; }
    [data-testid="stExpander"] { background-color: #ffffff !important; border: 1px solid #e5e5e5 !important; border-radius: 8px !important; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stApp { background-color: #ececec; }
    [data-testid="stSidebar"] { background-color: #ffffff; }
    h1, h2, h3, p, label, span { color: #1a1a1a !important; }
    [data-testid="stHeader"] { background-color: #f8f7f4 !important; }
    [data-testid="stSidebar"] .stButton > button { background-color: transparent !important; color: #1a1a1a !important; border: 1px solid #e5e5e5 !important; border-radius: 8px !important; }
</style>
""", unsafe_allow_html=True)

if "chats" not in st.session_state:
    st.session_state.chats = {"New Chat": []}
    st.session_state.current_chat = "New Chat"
if "history" not in st.session_state:
    st.session_state.history = st.session_state.chats[st.session_state.current_chat]
if "remaining" not in st.session_state:
    st.session_state.remaining = None
if st.session_state.get("dark_mode"):
    st.markdown("""
    <style>
    }
        section[data-testid="stBottom"] { background-color: #1a1a2e !important; border: none !important; }
        [data-testid="stSidebar"] * { color: #e0e0e0 !important; }
        [data-testid="stSidebar"] .stButton > button { color: #e0e0e0 !important; border-color: #533483 !important; }
        [data-testid="stHeader"] { background-color: #1a1a2e !important; }
        [data-testid="stExpander"] { background-color: #16213e !important; border-color: #533483 !important; }
        [data-testid="stChatInput"] { background-color: #0f3460 !important; border-color: #533483 !important; }
        [data-testid="stChatInputTextArea"] { background-color: #0f3460 !important; color: #e0e0e0 !important; }
        .stApp { background-color: #1a1a2e !important; }
        [data-testid="stSidebar"] { background-color: #16213e !important; }
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) { background-color: #0f3460 !important; border-color: #533483 !important; }
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) { background-color: #16213e !important; }
        section[data-testid="stBottom"] { background-color: #1a1a2e !important; }
        section[data-testid="stBottom"] > div { background-color: #1a1a2e !important; }
        [data-testid="stChatInput"] { background-color: #0f3460 !important; border-color: #533483 !important; }
        h1, h2, h3, p, label, span { color: #e0e0e0 !important; }
        .main * { color: #e0e0e0 !important; }
    </style>
    """, unsafe_allow_html=True)

model = "qwen2.5-0.5b-instruct-trtrtx-gpu:2"
fast_model = "qwen2.5-0.5b-instruct-trtrtx-gpu:2"

foundry_endpoint = None
known_ports = [63429, 58817, 54170, 53035, 61573, 63086, 57626, 5273, 57861, 8080]
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
    st.error("❌ Foundry Local is not running. Please start it with: `foundry model run phi-3.5-mini`")
    st.stop()

client = OpenAI(base_url=foundry_endpoint, api_key="local")

@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

embedder = load_embedder()

def detect_intent(question):
    q = question.lower()
    if any(word in q for word in ["summarize", "summary", "overview", "what is this document", "what does this document"]):
        return "SUMMARIZE"
    elif any(word in q for word in ["difference", "compare", "which one", "vs", "versus", "better", "both documents", "each document"]):
        return "COMPARE"
    elif any(word in q for word in ["calculate", "how many days", "convert", "translate"]):
        return "GENERAL"
    else:
        return "SEARCH"

with st.sidebar:
    st.image("logo.png", width=120)
    st.markdown("### Torqa")

    # Dark mode toggle
    if "dark_mode" not in st.session_state:
        st.session_state.dark_mode = False
    
    dark = st.toggle("🌙", value=st.session_state.dark_mode, key="dark_toggle")
    if dark != st.session_state.dark_mode:
        st.session_state.dark_mode = dark
        st.rerun()
    st.divider()

    if st.button("New Chat", use_container_width=True, key="new_chat_btn"):
        chat_count = len(st.session_state.chats) + 1
        new_chat_name = f"Chat {chat_count}"
        st.session_state.chats[new_chat_name] = []
        st.session_state.current_chat = new_chat_name
        st.session_state.history = st.session_state.chats[new_chat_name]
        st.rerun()

    st.markdown("**Chats**")
    for chat_name in list(st.session_state.chats.keys()):
        is_active = chat_name == st.session_state.current_chat
        col_a, col_b, col_c = st.columns([5, 4, 3])
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

st.markdown("<br>", unsafe_allow_html=True)

with st.expander("📎 Attach files"):
    uploaded_files = st.file_uploader("", type=["pdf", "txt"], accept_multiple_files=True, label_visibility="collapsed")

# GENERAL CHAT MODE
if not uploaded_files:
    for item in st.session_state.history:
        with st.chat_message("user"):
            st.write(item["question"])
        with st.chat_message("assistant"):
            st.write(item["answer"])

    if not st.session_state.history:
        c1, c2, c3 = st.columns([2, 1, 2])
        with c2:
                st.image("logo.png", width=200)

        st.markdown("<h1 style='text-align:center; color:#1a1a1a;'>Welcome to Torqa</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align:center; color:#666;'>Your local, offline AI document assistant</p>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        
        card1, card2, card3 = st.columns(3)
        with card1:
            st.markdown("""<div style="background:#ffffff;border:1px solid #e0e0e0;border-radius:12px;padding:20px;text-align:center;box-shadow:0 2px 8px rgba(0,0,0,0.06);">
                <div style="font-size:2rem;">📄</div>
                <div style="font-weight:bold;margin-top:8px;">Upload Documents</div>
                <div style="color:#666;font-size:0.9rem;margin-top:4px;">PDF or TXT files</div>
            </div>""", unsafe_allow_html=True)
        with card2:
            st.markdown("""<div style="background:#ffffff;border:1px solid #e0e0e0;border-radius:12px;padding:20px;text-align:center;box-shadow:0 2px 8px rgba(0,0,0,0.06);">
                <div style="font-size:2rem;">🔍</div>
                <div style="font-weight:bold;margin-top:8px;">Smart Search</div>
                <div style="color:#666;font-size:0.9rem;margin-top:4px;">BM25 + Embeddings</div>
            </div>""", unsafe_allow_html=True)
        with card3:
            st.markdown("""<div style="background:#ffffff;border:1px solid #e0e0e0;border-radius:12px;padding:20px;text-align:center;box-shadow:0 2px 8px rgba(0,0,0,0.06);">
                <div style="font-size:2rem;">🔒</div>
                <div style="font-weight:bold;margin-top:8px;">100% Offline</div>
                <div style="color:#666;font-size:0.9rem;margin-top:4px;">No internet needed</div>
            </div>""", unsafe_allow_html=True)

    general_question = st.chat_input("Ask anything or attach a document...", key="general_input")
    if general_question:
        with st.chat_message("user"):
            st.write(general_question)

        conversation_summary = ""
        if len(st.session_state.history) > 0:
            recent = st.session_state.history[-3:]
            conversation_summary = "Previous exchanges:\n" + "\n".join([f"Q: {item['question']}\nA: {item['answer']}" for item in recent]) + "\n\n"

        with st.chat_message("assistant"):
            answer_placeholder = st.empty()
            answer_placeholder.markdown("🤔 Torqa is thinking...")
            full_answer = ""
            import time
            start_time = time.time()
            stream = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": f"You are a helpful assistant. Answer in 1 sentence only. No notes, no explanations, just the answer.\n\n{conversation_summary}"},
                    {"role": "user", "content": general_question}
                ],
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
            elapsed = round(time.time() - start_time, 1)
            st.caption(f"⚡ Generated in {elapsed}s")
            if st.button("📋 Copy answer", key=f"copy_{hash(full_answer)}"):
                st.code(full_answer)
            answer = full_answer

        st.session_state.history.append({
            "question": general_question,
            "answer": answer,
            "confidence": 100,
            "sources": ["General Knowledge"],
            "timestamp": datetime.now()
        })
        st.session_state.chats[st.session_state.current_chat] = st.session_state.history

# DOCUMENT MODE
if uploaded_files:
    chunks = []
    sources = []

    for uploaded_file in uploaded_files:
        try:
            if uploaded_file.name.endswith(".pdf"):
                document = ""
                try:
                    with pdfplumber.open(uploaded_file) as pdf:
                        for page in pdf.pages:
                            text = page.extract_text()
                            if text:
                                document += text + "\n"
                            tables = page.extract_tables()
                            for table in tables:
                                for row in table:
                                    row_text = " | ".join([cell if cell else "" for cell in row])
                                    document += row_text + "\n"
                except Exception:
                    reader = PdfReader(uploaded_file)
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
    st.success(f"✅ {len(uploaded_files)} file(s) loaded — {total_words:,} words (~{estimated_pages} pages)")

    # Question suggestions
    file_key = "-".join([f.name for f in uploaded_files])
    if "suggestions" not in st.session_state or st.session_state.get("suggestions_key") != file_key:
        with st.spinner("Generating suggestions..."):
            suggestion_context = " ".join(chunks[:15])
            try:
                suggestion_response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "Based on this document, generate exactly 3 short questions a user might want to ask. Return ONLY the 3 questions, one per line, no numbering, no extra text."},
                        {"role": "user", "content": suggestion_context}
                    ]
                )
                st.session_state.suggestions = suggestion_response.choices[0].message.content.strip().split("\n")
                st.session_state.suggestions = [s.strip() for s in st.session_state.suggestions if s.strip()][:3]
                st.session_state.suggestions_key = file_key
            except Exception:
                st.session_state.suggestions = []

    if st.session_state.get("suggestions"):
        st.markdown("**💡 Suggested questions:**")
        cols = st.columns(len(st.session_state.suggestions))
        for i, suggestion in enumerate(st.session_state.suggestions):
            with cols[i]:
                if st.button(suggestion, key=f"suggestion_{i}", use_container_width=True):
                    st.session_state.selected_suggestion = suggestion

    if len(uploaded_files) >= 2:
        compare_mode = st.toggle("⚖️ Compare mode", help="Compare answers across documents side by side")
    else:
        compare_mode = False

    chunk_embeddings = embedder.encode(chunks)

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

    if "selected_suggestion" in st.session_state and st.session_state.selected_suggestion:
        question = st.session_state.selected_suggestion
        st.session_state.selected_suggestion = None
    else:
        question = st.chat_input("Ask about your documents...", key="doc_input")

    if question:
        with st.chat_message("user"):
            st.write(question)

        intent = detect_intent(question)

        if intent == "COMPARE" and len(uploaded_files) >= 2:
            compare_mode = True

        if intent == "SUMMARIZE":
            with st.chat_message("assistant"):
                answer_placeholder = st.empty()
                answer_placeholder.markdown("🤔 Torqa is thinking...")
                full_answer = ""
                import time
                start_time = time.time()
                try:
                    stream = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": "Summarize the following document clearly and concisely."},
                            {"role": "user", "content": " ".join(chunks[:10])}
                        ],
                        stream=True
                    )
                    try:
                        for chunk_part in stream:
                            if chunk_part.choices[0].delta.content is not None:
                                full_answer += chunk_part.choices[0].delta.content
                                answer_placeholder.markdown(full_answer + "▌")
                                
                    except Exception:
                        pass
                except Exception:
                    response = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": "Summarize the following document clearly and concisely."},
                            {"role": "user", "content": " ".join(chunks[:10])}
                        ]
                    )
                    full_answer = response.choices[0].message.content
                answer_placeholder.markdown(full_answer)
                elapsed = round(time.time() - start_time, 1)
                st.caption(f"⚡ Generated in {elapsed}s")
                if st.button("📋 Copy answer", key=f"copy_{hash(full_answer)}"):
                    st.code(full_answer)
                answer = full_answer

                confidence = 100
                top_sources = [f.name for f in uploaded_files]

            st.session_state.history.append({
                "question": question,
                "answer": answer,
                "confidence": confidence,
                "sources": list(set(top_sources)),
                "timestamp": datetime.now()
            })
            st.session_state.chats[st.session_state.current_chat] = st.session_state.history

        elif intent == "GENERAL":
            with st.chat_message("assistant"):
                answer_placeholder = st.empty()
                answer_placeholder.markdown("🤔 Torqa is thinking...")
                full_answer = ""
                import time
                start_time = time.time()
                stream = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant. Answer directly and concisely."},
                        {"role": "user", "content": question}
                    ],
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
                elapsed = round(time.time() - start_time, 1)
                st.caption(f"⚡ Generated in {elapsed}s")
                if st.button("📋 Copy answer", key=f"copy_{hash(full_answer)}"):
                    st.code(full_answer)
                answer = full_answer

            st.session_state.history.append({
                "question": question,
                "answer": answer,
                "confidence": 100,
                "sources": ["General Knowledge"],
                "timestamp": datetime.now()
            })
            st.session_state.chats[st.session_state.current_chat] = st.session_state.history

        else:  # SEARCH
            question_embedding = embedder.encode(question)

            if compare_mode and len(uploaded_files) >= 2:
                file_names = list(set(sources))
                cols = st.columns(len(file_names))
                answer = ""

                for idx, file_name in enumerate(file_names):
                    file_indices = [i for i, s in enumerate(sources) if s == file_name]
                    file_chunks = [chunks[i] for i in file_indices]
                    file_embeddings = np.array([chunk_embeddings[i] for i in file_indices])

                    file_dense_scores = np.dot(file_embeddings, question_embedding.flatten())
                    top_idx = np.argsort(file_dense_scores)[-2:][::-1]
                    file_context = "\n".join([file_chunks[i] for i in top_idx])

                    with cols[idx]:
                        st.markdown(f"**📄 {file_name}**")
                        col_placeholder = st.empty()
                        col_answer = ""

                        stream = client.chat.completions.create(
                            model=model,
                            messages=[
                                {"role": "system", "content": f"You are analyzing document: '{file_name}'. Answer the question based ONLY on this document's content. Be direct and concise in 2-3 sentences.\n\n{file_context}"},
                                {"role": "user", "content": question}
                            ],
                            stream=True
                        )
                        try:
                            for chunk_part in stream:
                                if chunk_part.choices[0].delta.content is not None:
                                    col_answer += chunk_part.choices[0].delta.content
                                    col_placeholder.markdown(col_answer + "▌")
                        except Exception:
                            pass
                        col_placeholder.markdown(col_answer)
                        answer += f"{file_name}: {col_answer}\n"

                confidence = 100
                top_sources = list(set(sources))

            else:
                dense_scores = np.dot(chunk_embeddings, question_embedding.flatten())
                dense_scores_norm = (dense_scores - dense_scores.min()) / (dense_scores.max() - dense_scores.min() + 1e-9)

                tokenized_chunks = [chunk.lower().split() for chunk in chunks]
                bm25 = BM25Okapi(tokenized_chunks)
                bm25_scores = bm25.get_scores(question.lower().split())
                bm25_scores_norm = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-9)

                hybrid_scores = 0.6 * dense_scores_norm + 0.4 * bm25_scores_norm
                top_indices = np.argsort(hybrid_scores)[-3:][::-1]
                top_chunks = [chunks[i] for i in top_indices]
                top_sources = [sources[i] for i in top_indices]
                context = "\n".join(list(top_chunks))
                confidence = int((float(hybrid_scores[top_indices[0]]) / float(np.max(hybrid_scores))) * 100)

                with st.chat_message("assistant"):
                    answer_placeholder = st.empty()
                    answer_placeholder.markdown("🤔 Torqa is thinking...")
                    full_answer = ""
                    import time
                    start_time = time.time()
                    stream = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": "Answer ONLY using the exact text below. The answer is in the text. Extract it directly.\n\nText:\n" + context},
                            {"role": "user", "content": question}
                        ],
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
                    elapsed = round(time.time() - start_time, 1)
                    st.caption(f"⚡ Generated in {elapsed}s")
                    if st.button("📋 Copy answer", key=f"copy_{hash(full_answer)}"):
                        st.code(full_answer)
                    answer = full_answer

                if confidence >= 75:
                    st.success(f"🟢 {confidence}% confidence — strong match found")
                elif confidence >= 50:
                    st.warning(f"🟡 {confidence}% confidence — partial match found")
                else:
                    st.error(f"🔴 {confidence}% confidence — weak match, answer may be inaccurate")

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
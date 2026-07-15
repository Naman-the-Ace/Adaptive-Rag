"""
Chat page for the Adaptive RAG Streamlit application.
Protected: redirects to Home if not authenticated.
"""

import streamlit as st
from utils.api_client import document_upload_rag, query_backend

# ── Auth guard ────────────────────────────────────────────────────────────────
if "jwt_token" not in st.session_state:
    st.switch_page("Home.py")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Adaptive RAG — Chat",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); }

    /* Sidebar glass */
    [data-testid="stSidebar"] {
        background: rgba(255,255,255,0.06) !important;
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255,255,255,0.1);
    }

    /* Chat messages */
    [data-testid="stChatMessage"] {
        background: rgba(255,255,255,0.07) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 14px !important;
        margin: 0.5rem 0 !important;
    }

    /* Chat input */
    [data-testid="stChatInput"] textarea {
        background: rgba(255,255,255,0.1) !important;
        border: 1px solid rgba(255,255,255,0.2) !important;
        border-radius: 12px !important;
        color: white !important;
    }

    /* Headers */
    h1, h2, h3 { color: white !important; }
    p, span, label { color: rgba(255,255,255,0.85) !important; }

    /* Upload area */
    [data-testid="stFileUploader"] {
        background: rgba(255,255,255,0.06) !important;
        border: 1px dashed rgba(255,255,255,0.3) !important;
        border-radius: 12px !important;
        padding: 1rem !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(102,126,234,0.4) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── State init ────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "show_logout_confirm" not in st.session_state:
    st.session_state.show_logout_confirm = False

token = st.session_state.get("jwt_token", "")
username = st.session_state.get("username", "User")
session_id = username  # Use username as stable session_id

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"### 👤 {username}")
    st.markdown("---")

    # Document upload
    st.markdown("### 📎 Upload Documents")
    st.caption("Upload PDFs or TXT files to add to the knowledge base.")
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["pdf", "txt"],
        key="doc_uploader",
        label_visibility="collapsed",
    )

    if uploaded_file and st.button("📤 Index Document"):
        with st.spinner(f"Indexing **{uploaded_file.name}**…"):
            result = document_upload_rag(
                uploaded_file.read(),
                uploaded_file.name,
                token,
            )
        if result.get("status") == "indexed":
            st.success(f"✅ Indexed {result['chunks']} chunks from `{result['filename']}`!")
        else:
            st.error(f"❌ {result.get('message', 'Unknown error')}")

    st.markdown("---")

    # Clear chat
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")

    # Logout
    if st.button("🔒 Logout"):
        st.session_state.show_logout_confirm = True

    if st.session_state.show_logout_confirm:
        st.warning("Are you sure you want to logout?")
        col_yes, col_no = st.columns(2)
        with col_yes:
            if st.button("✅ Yes"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.switch_page("Home.py")
        with col_no:
            if st.button("❌ No"):
                st.session_state.show_logout_confirm = False
                st.rerun()

# ── Main chat area ────────────────────────────────────────────────────────────
col_title, col_badge = st.columns([8, 2])
with col_title:
    st.markdown("# 🧠 Adaptive RAG Assistant")
with col_badge:
    st.markdown(
        """
        <div style="background:rgba(102,126,234,0.3);border:1px solid rgba(102,126,234,0.5);
            border-radius:20px;padding:6px 14px;text-align:center;margin-top:8px;">
            <span style="color:#a78bfa;font-size:0.8rem;font-weight:600;">● LIVE</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.caption("Ask anything — I'll search your documents, the web, or answer from general knowledge.")
st.markdown("---")

# ── Render existing messages ──────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "🧠"):
        st.markdown(msg["content"])

# ── Chat input ────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask me anything…"):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # Get AI response
    with st.chat_message("assistant", avatar="🧠"):
        with st.spinner("Thinking…"):
            answer = query_backend(prompt, token, session_id)
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})

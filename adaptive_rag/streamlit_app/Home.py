"""
Home page / Login & Registration UI for the Adaptive RAG Streamlit app.
Authenticates users against the FastAPI Auth Service using JWT Bearer tokens.
"""

import logging

import streamlit as st

from utils.api_client import login_user, register_user

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Adaptive RAG — Login",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Hide Streamlit default sidebar nav */
    [data-testid="stSidebarNav"] { display: none; }

    /* Dark gradient background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        min-height: 100vh;
    }

    /* Glass card */
    .login-card {
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 20px;
        padding: 2.5rem 2rem;
        margin: 2rem auto;
        max-width: 420px;
    }

    .login-title {
        font-size: 2rem;
        font-weight: 700;
        color: #ffffff;
        text-align: center;
        margin-bottom: 0.25rem;
    }

    .login-subtitle {
        font-size: 0.9rem;
        color: rgba(255,255,255,0.6);
        text-align: center;
        margin-bottom: 2rem;
    }

    /* Input fields */
    .stTextInput > div > div > input {
        background: rgba(255,255,255,0.1) !important;
        border: 1px solid rgba(255,255,255,0.2) !important;
        border-radius: 10px !important;
        color: white !important;
        padding: 0.75rem 1rem !important;
    }

    /* Buttons */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.75rem !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4) !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255,255,255,0.08) !important;
        border-radius: 10px !important;
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        color: rgba(255,255,255,0.7) !important;
        border-radius: 8px !important;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Skip login if already authenticated ──────────────────────────────────────
if "jwt_token" in st.session_state and "username" in st.session_state:
    st.switch_page("pages/chat.py")

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="login-card">
        <div class="login-title">🧠 Adaptive RAG</div>
        <div class="login-subtitle">Powered by Gemini · LangGraph · FastAPI</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Login / Register tabs ─────────────────────────────────────────────────────
tab_login, tab_register = st.tabs(["🔑 Login", "📝 Register"])

with tab_login:
    st.markdown("#### Sign in to your account")
    login_username = st.text_input("Username", key="login_user", placeholder="Enter your username")
    login_password = st.text_input("Password", key="login_pass", type="password", placeholder="Enter your password")

    if st.button("Login", key="btn_login"):
        if not login_username or not login_password:
            st.warning("Please enter both username and password.")
        else:
            with st.spinner("Authenticating…"):
                result = login_user(login_username, login_password)

            if result["success"]:
                st.session_state["jwt_token"] = result["token"]
                st.session_state["username"] = login_username
                st.success(f"Welcome back, **{login_username}**! Redirecting…")
                logger.info("User logged in: %s", login_username)
                st.switch_page("pages/chat.py")
            else:
                st.error(f"Login failed: {result['error']}")

with tab_register:
    st.markdown("#### Create a new account")
    reg_username = st.text_input("Username", key="reg_user", placeholder="Choose a username (min 3 chars)")
    reg_password = st.text_input("Password", key="reg_pass", type="password", placeholder="Choose a password (min 6 chars)")
    reg_password2 = st.text_input("Confirm Password", key="reg_pass2", type="password", placeholder="Repeat your password")

    if st.button("Create Account", key="btn_register"):
        if not reg_username or not reg_password:
            st.warning("Please fill in all fields.")
        elif reg_password != reg_password2:
            st.error("Passwords do not match.")
        elif len(reg_username) < 3:
            st.error("Username must be at least 3 characters.")
        elif len(reg_password) < 6:
            st.error("Password must be at least 6 characters.")
        else:
            with st.spinner("Creating account…"):
                result = register_user(reg_username, reg_password)
            if result["success"]:
                st.success("Account created! Please switch to the Login tab to sign in.")
                logger.info("New user registered: %s", reg_username)
            else:
                st.error(f"Registration failed: {result['error']}")

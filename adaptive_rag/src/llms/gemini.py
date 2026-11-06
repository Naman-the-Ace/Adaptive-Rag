"""
LLM and Embeddings configuration.
Supports Groq (ChatGroq) with local HuggingFace embeddings as primary,
with Gemini fallback.
"""

import os
from langchain_community.embeddings import HuggingFaceEmbeddings

# ── Local Embeddings (HuggingFace all-MiniLM-L6-v2 — 100% free, CPU local) ────
try:
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
except Exception:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model=os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001"),
        google_api_key=os.getenv("GEMINI_API_KEY"),
    )

# ── LLM Configuration ────────────────────────────────────────────────────────
groq_api_key = os.getenv("GROQ_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")

llm = None
if groq_api_key:
    from langchain_groq import ChatGroq
    llm = ChatGroq(
        model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        groq_api_key=groq_api_key,
        temperature=0.1,
    )
elif gemini_api_key:
    from langchain_google_genai import ChatGoogleGenerativeAI
    llm = ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
        google_api_key=gemini_api_key,
        temperature=0.1,
    )
else:
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
            temperature=0.1,
        )
    except Exception:
        llm = None

"""
Google Gemini LLM configuration for LangChain.
Replaces the original OpenAI integration (langchain_openai) with
langchain-google-genai for cost-effective, high-quality inference.
"""

import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# ── LLM (Gemini 1.5 Flash — fast and cost-effective) ─────────────────────────
llm = ChatGoogleGenerativeAI(
    model=os.getenv("GEMINI_MODEL", "gemini-1.5-flash"),
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.1,
    convert_system_message_to_human=True,
)

# ── Embeddings (text-embedding-004 — state-of-the-art from Google) ────────────
embeddings = GoogleGenerativeAIEmbeddings(
    model=os.getenv("GEMINI_EMBEDDING_MODEL", "models/text-embedding-004"),
    google_api_key=os.getenv("GEMINI_API_KEY"),
)

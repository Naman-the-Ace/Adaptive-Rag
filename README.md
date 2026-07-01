# 🧠 Adaptive RAG — Intelligent Document Q&A with Authentication

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.5-orange)](https://github.com/langchain-ai/langgraph)
[![Gemini](https://img.shields.io/badge/Gemini-1.5--Flash-4285F4?logo=google)](https://ai.google.dev)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.41-FF4B4B?logo=streamlit)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An **end-to-end Adaptive Retrieval-Augmented Generation (RAG)** system with a full authentication layer. Ask questions against your uploaded documents — the system intelligently decides whether to search your knowledge base, answer from general knowledge, or run a live web search.

---

## 📐 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Frontend                        │
│  ┌──────────────┐         ┌──────────────────────────────┐  │
│  │  Login Page  │  ────►  │  Chat UI + Document Upload   │  │
│  └──────────────┘         └──────────────────────────────┘  │
└──────────────────────────────┬──────────────────────────────┘
                               │ JWT Bearer Token
                               ▼
┌─────────────────────────────────────────────────────────────┐
│               Auth Service (FastAPI :8080)                  │
│  POST /auth/register  ·  POST /auth/login  ·  GET /user/me  │
│  POST /agent/chat  ──► proxies to RAG backend               │
│  SQLite (users.db) · Argon2id · JWT (python-jose)           │
└──────────────────────────────┬──────────────────────────────┘
                               │ Internal HTTP
                               ▼
┌─────────────────────────────────────────────────────────────┐
│          Adaptive RAG Backend (FastAPI :8000)               │
│                                                             │
│   POST /rag/query          POST /rag/documents/upload       │
│                                                             │
│   ┌─────────────────────────────────────────────────────┐  │
│   │              LangGraph Workflow                      │  │
│   │                                                      │  │
│   │  Query Classifier ─► [index] ─► FAISS Retriever     │  │
│   │         │                      │                     │  │
│   │         │                      ▼                     │  │
│   │         │              Document Grader               │  │
│   │         │              ↙           ↘                 │  │
│   │         │        Relevant      Irrelevant            │  │
│   │         │            │              │                │  │
│   │         ├─► [general]│              ▼                │  │
│   │         │   Gemini   │       Query Rewriter          │  │
│   │         │   Direct   │              │                │  │
│   │         │            │              ▼                │  │
│   │         └─► [search]─┴──────► Web Search (Tavily)   │  │
│   │                                    │                 │  │
│   │                                    ▼                 │  │
│   │                           Generate Answer (Gemini)   │  │
│   └─────────────────────────────────────────────────────┘  │
│   Gemini Embeddings (text-embedding-004) · FAISS · MongoDB  │
└─────────────────────────────────────────────────────────────┘
```

---

## ✨ Features

| Feature | Details |
|---------|---------|
| 🔐 **Authentication** | JWT Bearer tokens · Argon2id password hashing · Register/Login |
| 🧠 **Adaptive Routing** | Classifier decides: RAG / General LLM / Web Search |
| 📄 **Document Upload** | PDF & TXT ingestion → chunking → Gemini embedding → FAISS |
| 🌐 **Web Search Fallback** | Tavily API for real-time information |
| 💬 **Chat History** | Per-user session persistence in MongoDB |
| 🚀 **Gemini-Powered** | Gemini 1.5 Flash for LLM + text-embedding-004 for embeddings |
| 🖥️ **Streamlit UI** | Glassmorphism dark theme with login, chat, and file upload |

---

## 📁 Repository Structure

```
Adaptive-Rag/
├── adaptive_rag/                  # RAG backend + Streamlit frontend
│   ├── main.py                    # FastAPI entry point (:8000)
│   ├── requirements.txt
│   ├── src/
│   │   ├── api/routes.py          # /rag/query, /rag/documents/upload
│   │   ├── config/
│   │   │   ├── settings.py        # Environment config (Pydantic)
│   │   │   └── prompts.yaml       # LLM prompt templates
│   │   ├── llms/gemini.py         # Gemini LLM + embeddings
│   │   ├── memory/chat_history_mongo.py
│   │   ├── models/                # State, Grade, RouteIdentifier schemas
│   │   └── rag/
│   │       ├── graph_builder.py   # Full LangGraph workflow
│   │       ├── retriever_setup.py # FAISS vector store
│   │       └── document_upload.py # File ingestion pipeline
│   └── streamlit_app/
│       ├── Home.py                # Login / Register page
│       ├── pages/chat.py          # Protected chat UI
│       └── utils/api_client.py    # HTTP client (JWT Bearer)
│
├── auth_service/                  # Authentication BFF (migrated from Rust)
│   ├── app/
│   │   ├── main.py                # FastAPI entry point (:8080)
│   │   ├── auth.py                # JWT + Argon2id
│   │   ├── database.py            # Async SQLite (aiosqlite)
│   │   ├── models.py              # SQLAlchemy User model
│   │   ├── schemas.py             # Pydantic request/response schemas
│   │   ├── dependencies.py        # JWT bearer dependency
│   │   └── routers/
│   │       ├── auth.py            # POST /auth/register, /auth/login
│   │       ├── user.py            # GET /user/me
│   │       └── agent.py           # POST /agent/chat (proxy)
│   └── requirements.txt
│
├── .env.example                   # Environment variables template
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start (Local)

### Prerequisites

- Python 3.11+
- [Gemini API key](https://aistudio.google.com/app/apikey) (free)
- [Tavily API key](https://tavily.com) (free tier available)
- MongoDB (optional — chat history is disabled gracefully if unavailable)

### 1. Clone the repository

```bash
git clone https://github.com/itsmehotpants/Adaptive-Rag.git
cd Adaptive-Rag
```

### 2. Set up environment variables

```bash
cp .env.example .env
# Edit .env and fill in your GEMINI_API_KEY and TAVILY_API_KEY
```

### 3. Start the Auth Service

```bash
cd auth_service
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
```

Auth service will be available at: `http://localhost:8080`
API docs: `http://localhost:8080/docs`

### 4. Start the RAG Backend

```bash
cd adaptive_rag
python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

RAG backend at: `http://localhost:8000`
API docs: `http://localhost:8000/docs`

### 5. Start the Streamlit Frontend

```bash
cd adaptive_rag/streamlit_app
streamlit run Home.py
```

Frontend at: `http://localhost:8501`

---

## 🔑 Authentication Flow

```
1. Register:  POST /auth/register  { username, password }
2. Login:     POST /auth/login     { username, password }  →  { access_token }
3. Use token: Authorization: Bearer <access_token>
4. Chat:      POST /agent/chat     { query }  (proxied to RAG with session_id)
```

---

## 🌐 Deployment

### Option A: Streamlit Community Cloud (Frontend)

> **Note:** Streamlit Community Cloud runs only the Streamlit app. You need the backend services deployed separately.

1. Push your code to GitHub (already done).
2. Go to [share.streamlit.io](https://share.streamlit.io).
3. Click **New app** → Select `itsmehotpants/Adaptive-Rag`.
4. Set **Main file path**: `adaptive_rag/streamlit_app/Home.py`
5. Under **Advanced → Secrets**, add:
   ```toml
   AUTH_SERVICE_URL = "https://your-auth-service.onrender.com"
   RAG_SERVICE_URL = "https://your-rag-backend.onrender.com"
   ```
6. Click **Deploy**.

### Option B: Render.com (Backend Services — Free Tier)

#### Deploy Auth Service

1. Go to [render.com](https://render.com) → **New Web Service**.
2. Connect GitHub repo → Select `itsmehotpants/Adaptive-Rag`.
3. Settings:
   - **Root Directory**: `auth_service`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
4. Add environment variables: `JWT_SECRET`, `ACCESS_TOKEN_EXPIRE_MINUTES=60`

#### Deploy RAG Backend

1. **New Web Service** → same repo.
2. Settings:
   - **Root Directory**: `adaptive_rag`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
3. Add environment variables: `GEMINI_API_KEY`, `TAVILY_API_KEY`, `MONGODB_URL`

### Option C: Railway.app

```bash
# Install Railway CLI
npm install -g @railway/cli
railway login

# Deploy auth service
cd auth_service
railway up

# Deploy RAG backend
cd ../adaptive_rag
railway up
```

---

## 🔧 API Reference

### Auth Service (`:8080`)

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `POST` | `/auth/register` | ❌ | Create new user account |
| `POST` | `/auth/login` | ❌ | Login and receive JWT token |
| `GET` | `/user/me` | ✅ JWT | Get current user profile |
| `POST` | `/agent/chat` | ✅ JWT | Proxy chat to RAG backend |
| `GET` | `/health` | ❌ | Health check |

### RAG Backend (`:8000`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/rag/query` | Run adaptive RAG query |
| `POST` | `/rag/documents/upload` | Upload & index document |
| `GET` | `/` | Health check |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **LLM** | Google Gemini 1.5 Flash |
| **Embeddings** | Google text-embedding-004 |
| **RAG Orchestration** | LangGraph 0.5 |
| **Vector Store** | FAISS (in-memory) |
| **Web Search** | Tavily API |
| **Chat History** | MongoDB |
| **Auth Backend** | FastAPI + Argon2id + JWT |
| **Auth Database** | SQLite (aiosqlite) |
| **RAG Backend** | FastAPI + Uvicorn |
| **Frontend** | Streamlit |

---

## 📝 License

MIT © 2024 itsmehotpants

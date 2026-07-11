"""
FastAPI API routes for the Adaptive RAG backend.

Endpoints:
    POST /rag/query            – Run a query through the adaptive RAG pipeline
    POST /rag/documents/upload – Upload and index a PDF or TXT document
"""

import logging

from fastapi import APIRouter, File, Header, UploadFile
from langchain_core.messages import HumanMessage

from src.memory.chat_history_mongo import ChatHistory
from src.models.query_request import QueryRequest
from src.rag.document_upload import documents
from src.rag.graph_builder import builder

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/rag/query", summary="Run a RAG query")
async def rag_query(req: QueryRequest):
    """
    Process a user query through the adaptive RAG LangGraph pipeline.

    The pipeline:
    1. Classifies the query (index / general / search).
    2. Retrieves relevant documents if needed.
    3. Grades and optionally rewrites the query.
    4. Generates a final answer using Gemini.
    5. Persists the conversation to MongoDB.

    Args:
        req: QueryRequest with `query` string and `session_id`.

    Returns:
        JSON with the generated `answer` and the `session_id`.
    """
    chat_history = ChatHistory(session_id=req.session_id)
    history_messages = chat_history.get_history()

    # Build input state
    input_state = {
        "messages": history_messages + [HumanMessage(content=req.query)],
        "route": "",
        "latest_query": req.query,
        "documents": [],
        "generation": "",
        "session_id": req.session_id,
    }

    result = builder.invoke(input_state)
    answer = result.get("generation", "I could not generate an answer.")

    # Persist to MongoDB
    chat_history.add_message("human", req.query)
    chat_history.add_message("ai", answer)

    logger.info("Query answered for session '%s': %s…", req.session_id, answer[:80])
    return {"answer": answer, "session_id": req.session_id}


@router.post("/rag/documents/upload", summary="Upload and index a document")
async def upload_document(
    file: UploadFile = File(...),
    x_description: str = Header(default="", alias="X-Description"),
):
    """
    Upload a PDF or TXT document, split it into chunks, embed with Gemini,
    and index into the in-memory FAISS vector store.

    Args:
        file: The uploaded file (PDF or TXT).
        x_description: Optional description passed as X-Description header.

    Returns:
        Indexing result with chunk count and filename.
    """
    content = await file.read()
    result = await documents(content, file.filename, description=x_description)
    logger.info("Document '%s' indexed: %s", file.filename, result)
    return result

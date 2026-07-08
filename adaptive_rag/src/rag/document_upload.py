"""
Document upload and parsing module.
Handles PDF and TXT file ingestion, chunking, and FAISS indexing.
"""

import logging

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

from src.rag.retriever_setup import retriever_chain

logger = logging.getLogger(__name__)

# Text splitter configuration
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)


async def documents(file_content: bytes, filename: str, description: str = "") -> dict:
    """
    Process an uploaded document: load → split → embed → index.

    Args:
        file_content: Raw file bytes.
        filename: Original filename (used to determine file type).
        description: Optional description appended to document metadata.

    Returns:
        Dict with indexing result status and chunk count.
    """
    import tempfile
    import os

    # Write to temp file for loader compatibility
    suffix = ".pdf" if filename.lower().endswith(".pdf") else ".txt"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_content)
        tmp_path = tmp.name

    try:
        if suffix == ".pdf":
            loader = PyPDFLoader(tmp_path)
        else:
            loader = TextLoader(tmp_path, encoding="utf-8")

        raw_docs = loader.load()
        logger.info("Loaded %d pages/sections from '%s'.", len(raw_docs), filename)

        # Add metadata
        for doc in raw_docs:
            doc.metadata["source"] = filename
            if description:
                doc.metadata["description"] = description

        chunks = text_splitter.split_documents(raw_docs)
        logger.info("Split into %d chunks.", len(chunks))

        success = retriever_chain(chunks)
        if success:
            return {"status": "indexed", "chunks": len(chunks), "filename": filename}
        else:
            return {"status": "error", "message": "Failed to index documents."}
    finally:
        os.unlink(tmp_path)

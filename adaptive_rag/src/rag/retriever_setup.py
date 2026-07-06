"""
FAISS vector store retriever setup using Google Gemini embeddings.
Replaces the original OpenAIEmbeddings with GoogleGenerativeAIEmbeddings.

The FAISS index is held in memory and populated when documents are uploaded.
"""

import logging

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from src.llms.gemini import embeddings

logger = logging.getLogger(__name__)

# Module-level FAISS instance (populated on document upload)
_faiss_vectorstore: FAISS | None = None


def retriever_chain(chunks: list[Document]) -> bool:
    """
    Index a list of document chunks into FAISS using Gemini embeddings.

    Args:
        chunks: Document chunks to embed and store.

    Returns:
        True on success, False on failure.
    """
    global _faiss_vectorstore

    try:
        if _faiss_vectorstore is None:
            _faiss_vectorstore = FAISS.from_documents(chunks, embeddings)
            logger.info("Created new FAISS index with %d chunks.", len(chunks))
        else:
            _faiss_vectorstore.add_documents(chunks)
            logger.info("Added %d chunks to existing FAISS index.", len(chunks))
        return True
    except Exception as exc:
        logger.error("Failed to index documents: %s", exc)
        return False


def get_retriever():
    """
    Return a LangChain retriever from the current FAISS index.

    Returns:
        A LangChain VectorStoreRetriever, or None if no documents indexed.
    """
    if _faiss_vectorstore is None:
        logger.warning("No documents indexed yet. Returning empty retriever.")
        return None
    return _faiss_vectorstore.as_retriever(search_kwargs={"k": 4})

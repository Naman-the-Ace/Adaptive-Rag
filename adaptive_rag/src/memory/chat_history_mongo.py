"""
MongoDB chat history persistence for multi-turn conversations.
Stores and retrieves chat messages per session_id.
"""

import logging
import os

from pymongo import MongoClient
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

logger = logging.getLogger(__name__)

MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "adaptive_rag")


class ChatHistory:
    """Manages per-session chat history in MongoDB."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        try:
            self.client = MongoClient(MONGODB_URL, serverSelectionTimeoutMS=3000)
            self.db = self.client[MONGODB_DB_NAME]
            self.collection = self.db["chat_history"]
        except Exception as exc:
            logger.warning("MongoDB not available: %s. Chat history disabled.", exc)
            self.collection = None

    def add_message(self, role: str, content: str) -> None:
        """Persist a message to MongoDB."""
        if self.collection is None:
            return
        try:
            self.collection.insert_one({
                "session_id": self.session_id,
                "role": role,
                "content": content,
            })
        except Exception as exc:
            logger.error("Failed to save message: %s", exc)

    def get_history(self) -> list[BaseMessage]:
        """Retrieve chat history for this session from MongoDB."""
        if self.collection is None:
            return []
        try:
            docs = list(self.collection.find(
                {"session_id": self.session_id},
                {"_id": 0, "role": 1, "content": 1}
            ))
            messages = []
            for doc in docs:
                if doc["role"] == "human":
                    messages.append(HumanMessage(content=doc["content"]))
                else:
                    messages.append(AIMessage(content=doc["content"]))
            return messages
        except Exception as exc:
            logger.error("Failed to retrieve history: %s", exc)
            return []

    def clear(self) -> None:
        """Clear chat history for this session."""
        if self.collection is None:
            return
        try:
            self.collection.delete_many({"session_id": self.session_id})
        except Exception as exc:
            logger.error("Failed to clear history: %s", exc)

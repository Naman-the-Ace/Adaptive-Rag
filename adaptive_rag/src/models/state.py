"""
LangGraph state definition for the Adaptive RAG workflow.
"""

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class State(TypedDict):
    """
    Graph state shared across all nodes.

    Attributes:
        messages: Conversation history (accumulated with add_messages reducer).
        route: Classification result — 'index', 'general', or 'search'.
        latest_query: The current user query string.
        documents: Retrieved/searched documents for context.
        generation: Final generated answer string.
        session_id: User session identifier for MongoDB chat history.
    """

    messages: Annotated[list[BaseMessage], add_messages]
    route: str
    latest_query: str
    documents: list
    generation: str
    session_id: str

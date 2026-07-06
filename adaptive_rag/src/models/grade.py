"""
Pydantic models for structured LLM output in the RAG pipeline.
"""

from pydantic import BaseModel, Field


class RouteIdentifier(BaseModel):
    """Structured output for the query classifier node."""

    route: str = Field(
        description="The routing decision: 'index', 'general', or 'search'."
    )


class Grade(BaseModel):
    """Structured output for the document relevance grader node."""

    binary_score: str = Field(
        description="Relevance score: 'yes' if the document is relevant, 'no' otherwise."
    )


class VerificationResult(BaseModel):
    """Structured output for answer quality verification."""

    is_supported: str = Field(
        description="'yes' if the answer is supported by the context, 'no' otherwise."
    )
    is_useful: str = Field(
        description="'yes' if the answer is useful to the user, 'no' otherwise."
    )

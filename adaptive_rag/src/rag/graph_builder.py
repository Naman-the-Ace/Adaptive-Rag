"""
LangGraph graph builder for the Adaptive RAG workflow.
Uses Google Gemini LLM (via langchain-google-genai).

Graph flow:
    START → query_classifier
        → [index]   retrieve_docs → grade_docs → [relevant] generate → END
                                               → [irrelevant] rewrite → web_search → generate → END
        → [general] general_llm → END
        → [search]  web_search → generate → END
"""

import logging

from langchain_community.tools import TavilySearchResults
from langchain_core.messages import AIMessage
from langchain_core.prompts import PromptTemplate
from langgraph.constants import END, START
from langgraph.graph.state import StateGraph

from src.config.settings import Config
from src.llms.gemini import llm
from src.models.grade import Grade, RouteIdentifier
from src.models.state import State
from src.rag.retriever_setup import get_retriever

logger = logging.getLogger(__name__)
config = Config()

# ── Tavily web search tool ────────────────────────────────────────────────────
tavily_tool = TavilySearchResults(max_results=3)


# ══════════════════════════════════════════════════════════════════════════════
# NODE IMPLEMENTATIONS
# ══════════════════════════════════════════════════════════════════════════════


def query_classifier(state: State) -> dict:
    """
    Classify the query: 'index', 'general', or 'search'.
    Uses Qdrant/FAISS context to determine if indexed docs can answer.
    """
    question = state["messages"][-1].content
    retriever = get_retriever()

    context = []
    if retriever:
        context = retriever.invoke(question)

    llm_structured = llm.with_structured_output(RouteIdentifier)
    classify_prompt = PromptTemplate(
        template=config.prompt("classify_prompt"),
        input_variables=["question", "context"],
    )
    chain = classify_prompt | llm_structured
    result = chain.invoke({"question": question, "context": context})
    logger.info("Query classified as: %s", result.route)

    return {
        "messages": state["messages"],
        "route": result.route,
        "latest_query": question,
        "documents": context,
        "generation": "",
        "session_id": state.get("session_id", "default"),
    }


def retrieve_docs(state: State) -> dict:
    """Retrieve relevant documents from FAISS for the current query."""
    retriever = get_retriever()
    docs = retriever.invoke(state["latest_query"]) if retriever else []
    logger.info("Retrieved %d documents.", len(docs))
    return {**state, "documents": docs}


def grade_docs(state: State) -> dict:
    """
    Grade each retrieved document for relevance.
    Filters out irrelevant documents; sets web_search flag if none remain.
    """
    question = state["latest_query"]
    docs = state["documents"]

    llm_structured = llm.with_structured_output(Grade)
    grade_prompt = PromptTemplate(
        template=config.prompt("grade_prompt"),
        input_variables=["question", "document"],
    )
    chain = grade_prompt | llm_structured

    relevant_docs = []
    for doc in docs:
        result = chain.invoke({"question": question, "document": doc.page_content})
        if result.binary_score == "yes":
            relevant_docs.append(doc)

    logger.info("%d/%d documents passed relevance grading.", len(relevant_docs), len(docs))
    return {**state, "documents": relevant_docs}


def rewrite_query(state: State) -> dict:
    """Rewrite the query to improve retrieval when no relevant docs found."""
    rewrite_prompt = PromptTemplate(
        template=config.prompt("rewrite_prompt"),
        input_variables=["question"],
    )
    chain = rewrite_prompt | llm
    result = chain.invoke({"question": state["latest_query"]})
    new_query = result.content if hasattr(result, "content") else str(result)
    logger.info("Query rewritten: %s → %s", state["latest_query"], new_query)
    return {**state, "latest_query": new_query}


def web_search(state: State) -> dict:
    """Run Tavily web search for real-time or niche queries."""
    query = state["latest_query"]
    results = tavily_tool.invoke({"query": query})

    from langchain_core.documents import Document

    docs = [
        Document(page_content=r["content"], metadata={"source": r["url"]})
        for r in results
        if "content" in r
    ]
    logger.info("Web search returned %d results.", len(docs))
    return {**state, "documents": docs}


def generate_answer(state: State) -> dict:
    """Generate a final answer using retrieved/searched documents as context."""
    question = state["latest_query"]
    docs = state["documents"]
    context = "\n\n".join(doc.page_content for doc in docs)

    generate_prompt = PromptTemplate(
        template=config.prompt("generate_prompt"),
        input_variables=["context", "question"],
    )
    chain = generate_prompt | llm
    result = chain.invoke({"context": context, "question": question})
    generation = result.content if hasattr(result, "content") else str(result)

    return {
        **state,
        "generation": generation,
        "messages": state["messages"] + [AIMessage(content=generation)],
    }


def general_llm_answer(state: State) -> dict:
    """Answer general/casual questions directly using Gemini without retrieval."""
    question = state["latest_query"]
    result = llm.invoke(question)
    generation = result.content if hasattr(result, "content") else str(result)
    logger.info("General LLM answered: %s…", generation[:80])
    return {
        **state,
        "generation": generation,
        "messages": state["messages"] + [AIMessage(content=generation)],
    }


# ══════════════════════════════════════════════════════════════════════════════
# CONDITIONAL EDGES
# ══════════════════════════════════════════════════════════════════════════════


def route_query(state: State) -> str:
    """Route after classification."""
    return state["route"]  # 'index', 'general', or 'search'


def route_after_grading(state: State) -> str:
    """Route after document grading: generate if relevant docs exist, else rewrite."""
    return "generate" if state["documents"] else "rewrite"


# ══════════════════════════════════════════════════════════════════════════════
# GRAPH COMPILATION
# ══════════════════════════════════════════════════════════════════════════════

workflow = StateGraph(State)

# Add nodes
workflow.add_node("query_classifier", query_classifier)
workflow.add_node("retrieve_docs", retrieve_docs)
workflow.add_node("grade_docs", grade_docs)
workflow.add_node("rewrite_query", rewrite_query)
workflow.add_node("web_search", web_search)
workflow.add_node("generate_answer", generate_answer)
workflow.add_node("general_llm_answer", general_llm_answer)

# Entry point
workflow.add_edge(START, "query_classifier")

# Conditional routing after classification
workflow.add_conditional_edges(
    "query_classifier",
    route_query,
    {
        "index": "retrieve_docs",
        "general": "general_llm_answer",
        "search": "web_search",
    },
)

# Index path
workflow.add_edge("retrieve_docs", "grade_docs")
workflow.add_conditional_edges(
    "grade_docs",
    route_after_grading,
    {
        "generate": "generate_answer",
        "rewrite": "rewrite_query",
    },
)
workflow.add_edge("rewrite_query", "web_search")

# Converge all answer paths
workflow.add_edge("web_search", "generate_answer")
workflow.add_edge("generate_answer", END)
workflow.add_edge("general_llm_answer", END)

# Compile
builder = workflow.compile()

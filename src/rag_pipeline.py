import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from langchain_groq import ChatGroq

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.hybrid import hybrid_retriever
from src.prompts import SYSTEM_PROMPT_1, SYSTEM_PROMPT_2, SYSTEM_PROMPT_3, build_prompt
from src.semantic import get_embedding_model, load_semantic_artifacts, semantic_search

GROQ_MODEL_NAME = "llama-3.1-8b-instant"
from langchain_community.tools import DuckDuckGoSearchRun

load_dotenv()
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)
web_search_tool = DuckDuckGoSearchRun()

TOP_K = 5
TEST_QUERY = "romantic comedy movie about weddings"


def load_llm() -> ChatGroq:
    """Load and return the Groq LLM used in the RAG pipeline."""
    load_dotenv()
    return ChatGroq(
        model=GROQ_MODEL_NAME,
        api_key=os.getenv("GROQ_API_KEY"),
    )


def get_semantic_retriever() -> tuple:
    """Load and return the semantic retriever components."""
    model = get_embedding_model()
    index, documents = load_semantic_artifacts()
    return model, index, documents


def retrieve_semantic(query: str, top_k: int = TOP_K) -> pd.DataFrame:
    """Retrieve the top-k semantic search results for a query."""
    model, index, documents = get_semantic_retriever()
    results = semantic_search(
        query=query,
        index=index,
        documents=documents,
        model=model,
        top_k=top_k,
    )
    return results


def build_context(docs_df: pd.DataFrame) -> str:
    """Format retrieved documents into a context block for the LLM."""
    context_blocks = []
    for i, (_, row) in enumerate(docs_df.iterrows(), 1):
        asin = row.get("parent_asin", "N/A")
        title = row.get("product_title", "Unknown Title")
        rating = row.get("rating", "N/A")
        body = row.get("text", "")
        block = f"[{i}] {title} (ASIN: {asin}) - Rating: {rating}/5\nReview: {body}"
        context_blocks.append(block)
    return "\n\n".join(context_blocks)


def run_rag_pipeline(
    query: str,
    top_k: int = TOP_K,
    system_prompt: str = SYSTEM_PROMPT_1,
) -> tuple[str, pd.DataFrame]:
    """Run the semantic RAG pipeline and return the answer with retrieved documents."""
    llm = load_llm()
    results_df = retrieve_semantic(query, top_k=top_k)
    context_str = build_context(results_df)
    final_prompt = build_prompt(query, context_str, system_prompt=system_prompt)
    response = llm.invoke(final_prompt)
    answer = response.content

    return answer, results_df


def run_hybrid_rag_pipeline(
    query: str,
    top_k: int = TOP_K,
    system_prompt: str = SYSTEM_PROMPT_1,
) -> tuple[str, pd.DataFrame]:
    """Run the hybrid RAG pipeline and return the answer with retrieved documents."""
    llm = load_llm()
    results_df = hybrid_retriever(query, top_k=top_k)
    context_str = build_context(results_df)
    final_prompt = build_prompt(query, context_str, system_prompt=system_prompt)
    response = llm.invoke(final_prompt)
    answer = response.content

    return answer, results_df

def run_pipeline_with_tool(query: str, system_prompt):
    answer, retrieved_docs = run_hybrid_rag_pipeline(query, system_prompt=system_prompt)
    answer_text = answer.content if hasattr(answer, 'content') else str(answer)
    failure_phrases = ["cannot find", "more information is needed", "none of the reviews", "not available", "couldn't find"]
    
    if any(phrase in answer_text.lower() for phrase in failure_phrases):
        print(f"\n[Tool Triggered]: Amazon data insufficient for '{query}'. Searching the web...")
        web_results = web_search_tool.run(query)
        tool_prompt = (
            f"You are a helpful assistant. Based on the following live web search data, "
            f"please answer the user's query.\n\n"
            f"Web Data: {web_results}\n\n"
            f"User Query: {query}"
        )
        
        final_answer = llm.invoke(tool_prompt)
        final_answer_text = final_answer.content if hasattr(final_answer, 'content') else str(final_answer)
        return final_answer_text, "Web Search Used"
    
    return answer_text, "Hybrid RAG Used"

def main() -> None:
    """Run a simple test for the semantic and hybrid RAG pipelines."""
    print("Testing Semantic RAG Pipeline...\n")
    semantic_answer, semantic_docs = run_rag_pipeline(
        TEST_QUERY,
        system_prompt=SYSTEM_PROMPT_1,
    )
    print("\n===== SEMANTIC RAG ANSWER =====")
    print(semantic_answer)
    print("\n===== SEMANTIC RETRIEVED DOCS =====")
    print(semantic_docs[["rank", "product_title", "source"]].head())

    print("\n\nTesting Hybrid RAG Pipeline...\n")
    hybrid_answer, hybrid_docs = run_hybrid_rag_pipeline(
        TEST_QUERY,
        system_prompt=SYSTEM_PROMPT_1,
    )
    print("\n===== HYBRID RAG ANSWER =====")
    print(hybrid_answer)
    print("\n===== HYBRID RETRIEVED DOCS =====")
    print(hybrid_docs[["rank", "product_title", "source"]].head())


if __name__ == "__main__":
    main()
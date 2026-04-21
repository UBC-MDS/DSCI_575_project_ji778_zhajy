import sys
import os
from pathlib import Path
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
from langchain_groq import ChatGroq
from src.prompts import build_prompt, SYSTEM_PROMPT_1, SYSTEM_PROMPT_2, SYSTEM_PROMPT_3
from src.semantic import (
    get_embedding_model,
    load_semantic_artifacts,
    semantic_search,
)
from src.hybrid import hybrid_retriever
from langchain_community.tools import DuckDuckGoSearchRun

load_dotenv()
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)
web_search_tool = DuckDuckGoSearchRun()

TOP_K = 5

def get_semantic_retriever():
    model = get_embedding_model()
    index, documents = load_semantic_artifacts()
    return model, index, documents

def retrieve_semantic(query: str, top_k: int = TOP_K) -> pd.DataFrame:
    model, index, documents = get_semantic_retriever()
    results = semantic_search(
        query=query, index=index, documents=documents, model=model, top_k=top_k,
    )
    return results

def build_context(docs_df):
    context_blocks = []
    for i, (_, row) in enumerate(docs_df.iterrows(), 1):
        asin = row.get('parent_asin', 'N/A')
        title = row.get('product_title', 'Unknown Title')
        rating = row.get('rating', 'N/A')
        body = row.get('text', '')
        block = f"[{i}] {title} (ASIN: {asin}) - Rating: {rating}/5\nReview: {body}"
        context_blocks.append(block)
    return "\n\n".join(context_blocks)

def run_rag_pipeline(query: str, top_k: int = TOP_K, system_prompt=SYSTEM_PROMPT_1):
    results_df = retrieve_semantic(query, top_k=top_k)
    context_str = build_context(results_df)
    final_prompt = build_prompt(query, context_str, system_prompt=system_prompt)
    response = llm.invoke(final_prompt)
    answer = response.content 
    
    return answer, results_df

def run_hybrid_rag_pipeline(query: str, top_k: int = TOP_K, system_prompt=SYSTEM_PROMPT_1):
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

if __name__ == "__main__":
    test_query = "romantic comedy movie about weddings"

    print("Testing Semantic RAG Pipeline...\n")
    semantic_answer, semantic_docs = run_rag_pipeline(
        test_query, system_prompt=SYSTEM_PROMPT_1
    )
    print("\n===== SEMANTIC RAG ANSWER =====")
    print(semantic_answer)
    print("\n===== SEMANTIC RETRIEVED DOCS =====")
    print(semantic_docs[["rank", "product_title", "source"]].head())

    print("\n\nTesting Hybrid RAG Pipeline...\n")
    hybrid_answer, hybrid_docs = run_hybrid_rag_pipeline(
        test_query, system_prompt=SYSTEM_PROMPT_1
    )
    print("\n===== HYBRID RAG ANSWER =====")
    print(hybrid_answer)
    print("\n===== HYBRID RETRIEVED DOCS =====")
    print(hybrid_docs[["rank", "product_title", "source"]].head())
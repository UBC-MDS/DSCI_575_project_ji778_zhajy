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

load_dotenv()
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)

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
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

if __name__ == "__main__":
    test_query = "romantic comedy movie about weddings"
    
    print("Testing Pipeline with Groq Llama 3.1...\n")
    
    ans1, _ = run_rag_pipeline(test_query, system_prompt=SYSTEM_PROMPT_1)
    print("\n===== EXPERIMENT 1: The Precise Critic =====")
    print(ans1)
    
    ans2, _ = run_rag_pipeline(test_query, system_prompt=SYSTEM_PROMPT_2)
    print("\n===== EXPERIMENT 2: The Community Guide =====")
    print(ans2)
    
    ans3, _ = run_rag_pipeline(test_query, system_prompt=SYSTEM_PROMPT_3)
    print("\n===== EXPERIMENT 3: The Fact-Checker =====")
    print(ans3)
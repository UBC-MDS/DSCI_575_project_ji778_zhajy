import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
from src.semantic import (
    get_embedding_model,
    load_semantic_artifacts,
    semantic_search,
)

TOP_K = 5


def get_semantic_retriever():
    model = get_embedding_model()
    index, documents = load_semantic_artifacts()
    return model, index, documents


def retrieve_semantic(query: str, top_k: int = TOP_K) -> pd.DataFrame:
    model, index, documents = get_semantic_retriever()

    results = semantic_search(
        query=query,
        index=index,
        documents=documents,
        model=model,
        top_k=top_k,
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

if __name__ == "__main__":
    query = "romantic comedy movie"
    results = retrieve_semantic(query, top_k=5)
    print(results[["rank", "score", "product_title", "text"]].head())
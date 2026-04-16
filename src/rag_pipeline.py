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


if __name__ == "__main__":
    query = "romantic comedy movie"
    results = retrieve_semantic(query, top_k=5)
    print(results[["rank", "score", "product_title", "text"]].head())
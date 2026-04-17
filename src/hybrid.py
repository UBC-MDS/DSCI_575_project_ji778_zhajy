import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
from src.bm25 import BM25Retriever
from src.semantic import (
    get_embedding_model,
    load_semantic_artifacts,
    semantic_search,
)

TOP_K = 5


def get_bm25_retriever():
    bm25 = BM25Retriever(
        index_path="data/processed/bm25_index.pkl",
        corpus_path="data/processed/corpus_data.pkl",
    )
    bm25.load_index()
    return bm25


def retrieve_bm25(query: str, top_k: int = TOP_K) -> pd.DataFrame:
    bm25 = get_bm25_retriever()
    results = bm25.search(query, top_n=top_k)

    rows = []
    for rank, (doc, score) in enumerate(results, start=1):
        rows.append(
            {
                "rank": rank,
                "score": float(score),
                "parent_asin": doc.get("parent_asin", "N/A"),
                "product_title": doc.get("product_title", "Unknown Title"),
                "rating": doc.get("rating", "N/A"),
                "text": doc.get("review_text", ""),
                "source": "bm25",
            }
        )

    return pd.DataFrame(rows)


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


def hybrid_retriever(query: str, top_k: int = TOP_K, rrf_k: int = 60) -> pd.DataFrame:
    bm25_df = retrieve_bm25(query, top_k=top_k).copy()
    semantic_df = retrieve_semantic(query, top_k=top_k).copy()

    if "source" not in semantic_df.columns:
        semantic_df["source"] = "semantic"

    bm25_df["bm25_rank"] = range(1, len(bm25_df) + 1)
    semantic_df["semantic_rank"] = range(1, len(semantic_df) + 1)

    bm25_df["dedup_key"] = (
        bm25_df["parent_asin"].astype(str) + "||" + bm25_df["text"].astype(str)
    )
    semantic_df["dedup_key"] = (
        semantic_df["parent_asin"].astype(str) + "||" + semantic_df["text"].astype(str)
    )

    combined = pd.concat([bm25_df, semantic_df], ignore_index=True)

    fused_rows = []
    for _, group in combined.groupby("dedup_key", sort=False):
        row = group.iloc[0].copy()

        bm25_rank = group["bm25_rank"].dropna().min() if "bm25_rank" in group.columns else None
        semantic_rank = group["semantic_rank"].dropna().min() if "semantic_rank" in group.columns else None

        rrf_score = 0.0
        if pd.notna(bm25_rank):
            rrf_score += 1 / (rrf_k + bm25_rank)
        if pd.notna(semantic_rank):
            rrf_score += 1 / (rrf_k + semantic_rank)

        sources = group["source"].dropna().unique().tolist()
        row["source"] = "+".join(sorted(sources))
        row["score"] = rrf_score
        fused_rows.append(row)

    fused_df = pd.DataFrame(fused_rows)
    fused_df = fused_df.sort_values("score", ascending=False).head(top_k).reset_index(drop=True)
    fused_df["rank"] = range(1, len(fused_df) + 1)

    cols = ["rank", "score", "product_title", "text", "rating", "parent_asin", "source"]
    existing_cols = [c for c in cols if c in fused_df.columns]

    return fused_df[existing_cols]


if __name__ == "__main__":
    query = "romantic comedy movie about weddings"

    print("===== BM25 =====")
    print(retrieve_bm25(query, top_k=5)[["rank", "score", "product_title", "text"]].head())

    print("\n===== SEMANTIC =====")
    print(retrieve_semantic(query, top_k=5)[["rank", "score", "product_title", "text"]].head())

    print("\n===== HYBRID =====")
    print(hybrid_retriever(query, top_k=5)[["rank", "score", "product_title", "text", "source"]].head())
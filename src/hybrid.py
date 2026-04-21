import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.bm25 import BM25Retriever
from src.semantic import (
    get_embedding_model,
    load_semantic_artifacts,
    semantic_search,
)

DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"

BM25_INDEX_PATH = PROCESSED_DIR / "bm25_index.pkl"
BM25_CORPUS_PATH = PROCESSED_DIR / "corpus_data.pkl"

TOP_K = 5
RRF_K = 60
TEST_QUERY = "romantic comedy movie about weddings"


def get_bm25_retriever() -> BM25Retriever:
    """Load and return the BM25 retriever with saved artifacts."""
    bm25 = BM25Retriever(
        index_path=str(BM25_INDEX_PATH),
        corpus_path=str(BM25_CORPUS_PATH),
    )
    bm25.load_index()
    return bm25


def retrieve_bm25(query: str, top_k: int = TOP_K) -> pd.DataFrame:
    """Retrieve the top-k BM25 results for a query as a DataFrame."""
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


def get_semantic_retriever() -> tuple:
    """Load and return the semantic retriever components."""
    model = get_embedding_model()
    index, documents = load_semantic_artifacts()
    return model, index, documents


def retrieve_semantic(query: str, top_k: int = TOP_K) -> pd.DataFrame:
    """Retrieve the top-k semantic results for a query as a DataFrame."""
    model, index, documents = get_semantic_retriever()
    results = semantic_search(
        query=query,
        index=index,
        documents=documents,
        model=model,
        top_k=top_k,
    )
    return results


def hybrid_retriever(
    query: str,
    top_k: int = TOP_K,
    rrf_k: int = RRF_K,
) -> pd.DataFrame:
    """Combine BM25 and semantic results using Reciprocal Rank Fusion."""
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

        bm25_rank = group["bm25_rank"].dropna().min()
        semantic_rank = group["semantic_rank"].dropna().min()

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


def main() -> None:
    """Run a simple test of BM25, semantic, and hybrid retrieval."""
    print("===== BM25 =====")
    print(retrieve_bm25(TEST_QUERY, top_k=5)[["rank", "score", "product_title", "text"]].head())

    print("\n===== SEMANTIC =====")
    print(retrieve_semantic(TEST_QUERY, top_k=5)[["rank", "score", "product_title", "text"]].head())

    print("\n===== HYBRID =====")
    print(hybrid_retriever(TEST_QUERY, top_k=5)[["rank", "score", "product_title", "text", "source"]].head())


if __name__ == "__main__":
    main()
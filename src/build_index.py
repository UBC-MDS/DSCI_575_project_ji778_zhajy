import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
from src.bm25 import BM25Retriever

REVIEWS_PATH = "data/processed/reviews_retrieval_processed.csv"
META_PATH = "data/processed/meta_retrieval_processed.csv"
INDEX_PATH = "data/processed/bm25_index.pkl"
CORPUS_PATH = "data/processed/corpus_data.pkl"


def build_corpus_metadata(
    reviews_path: str = REVIEWS_PATH,
    meta_path: str = META_PATH,
) -> list[dict]:
    reviews = pd.read_csv(reviews_path)
    meta = pd.read_csv(meta_path)

    reviews = reviews.copy()
    meta = meta.copy()

    meta_subset = meta[["parent_asin", "title"]].copy()
    meta_subset = meta_subset.rename(columns={"title": "product_title"})

    merged = reviews.merge(meta_subset, on="parent_asin", how="inner")

    merged["product_title"] = merged["product_title"].fillna("").astype(str).str.strip()
    merged["title"] = merged["title"].fillna("").astype(str).str.strip()   # review title
    merged["text"] = merged["text"].fillna("").astype(str).str.strip()

    merged["retrieval_text"] = (
        merged["product_title"] + ". "
        + merged["title"] + ". "
        + merged["text"]
    ).str.replace(r"\s+", " ", regex=True).str.strip()

    merged = merged[
        (merged["product_title"].str.len() > 0) &
        (merged["retrieval_text"].str.len() > 0)
    ].reset_index(drop=True)

    corpus_metadata = []
    for _, row in merged.iterrows():
        corpus_metadata.append(
            {
                "parent_asin": row["parent_asin"],
                "product_title": row["product_title"],
                "review_text": row["text"],
                "rating": row["rating"],
                "retrieval_text": row["retrieval_text"],
            }
        )

    return corpus_metadata


def main():
    corpus_metadata = build_corpus_metadata()

    print(f"Number of BM25 documents: {len(corpus_metadata)}")
    if corpus_metadata:
        print("Example keys:", corpus_metadata[0].keys())

    bm25 = BM25Retriever(
        index_path=INDEX_PATH,
        corpus_path=CORPUS_PATH,
    )
    bm25.build_and_save_index(corpus_metadata)

    print(f"Saved BM25 index to {INDEX_PATH}")
    print(f"Saved corpus metadata to {CORPUS_PATH}")


if __name__ == "__main__":
    main()
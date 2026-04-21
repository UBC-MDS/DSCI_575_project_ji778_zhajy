import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.bm25 import BM25Retriever

DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"

REVIEWS_PATH = PROCESSED_DIR / "reviews_retrieval_processed.csv"
META_PATH = PROCESSED_DIR / "meta_retrieval_processed.csv"
INDEX_PATH = PROCESSED_DIR / "bm25_index.pkl"
CORPUS_PATH = PROCESSED_DIR / "corpus_data.pkl"


def build_corpus_metadata(
    reviews_path: str | Path = REVIEWS_PATH,
    meta_path: str | Path = META_PATH,
) -> list[dict]:
    """Build BM25 corpus metadata from processed reviews and metadata files."""
    reviews = pd.read_csv(reviews_path)
    meta = pd.read_csv(meta_path)

    reviews = reviews.copy()
    meta = meta.copy()

    meta_subset = meta[["parent_asin", "title"]].copy()
    meta_subset = meta_subset.rename(columns={"title": "product_title"})

    merged = reviews.merge(meta_subset, on="parent_asin", how="inner")

    merged["product_title"] = merged["product_title"].fillna("").astype(str).str.strip()
    merged["title"] = merged["title"].fillna("").astype(str).str.strip()
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


def main() -> None:
    """Build and save the BM25 index and corpus metadata artifacts."""
    corpus_metadata = build_corpus_metadata()

    print(f"Number of BM25 documents: {len(corpus_metadata)}")
    if corpus_metadata:
        print("Example keys:", corpus_metadata[0].keys())

    bm25 = BM25Retriever(
        index_path=str(INDEX_PATH),
        corpus_path=str(CORPUS_PATH),
    )
    bm25.build_and_save_index(corpus_metadata)

    print(f"Saved BM25 index to {INDEX_PATH}")
    print(f"Saved corpus metadata to {CORPUS_PATH}")


if __name__ == "__main__":
    main()
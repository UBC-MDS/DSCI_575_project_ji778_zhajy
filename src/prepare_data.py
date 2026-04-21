import gzip
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

REVIEW_PATH = RAW_DIR / "Movies_and_TV.jsonl.gz"
META_PATH = RAW_DIR / "meta_Movies_and_TV.jsonl.gz"

TARGET_PRODUCTS = 10000
MAX_REVIEWS_PER_PRODUCT = 1


def build_metadata_lookup(meta_path: str | Path) -> dict:
    """
    Build a lookup of parent_asin to metadata record,
    keeping only metadata rows with a non-empty title.
    """
    meta_lookup = {}

    with gzip.open(meta_path, "rt", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)

            parent_asin = str(record.get("parent_asin", "")).strip()
            raw_title = record.get("title", "")

            title = ""
            if raw_title is not None:
                title = str(raw_title).strip()

            if parent_asin and title:
                meta_lookup[parent_asin] = record

    return meta_lookup


def collect_matched_reviews(
    review_path: str | Path,
    meta_lookup: dict,
    target_products: int,
    max_reviews_per_product: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Collect review rows until at least `target_products` unique parent_asin values
    are included. Optionally limit the number of reviews kept per product.
    """
    matched_reviews = []
    matched_asins = set()
    review_counts = defaultdict(int)

    with gzip.open(review_path, "rt", encoding="utf-8") as f:
        for line in f:
            review = json.loads(line)
            parent_asin = str(review.get("parent_asin", "")).strip()

            if parent_asin not in meta_lookup:
                continue

            if review_counts[parent_asin] >= max_reviews_per_product:
                continue

            matched_reviews.append(review)
            matched_asins.add(parent_asin)
            review_counts[parent_asin] += 1

            if len(matched_asins) >= target_products:
                break

    matched_meta = [meta_lookup[asin] for asin in matched_asins]

    reviews_df = pd.DataFrame(matched_reviews)
    meta_df = pd.DataFrame(matched_meta)

    return reviews_df, meta_df


def preprocess_reviews(reviews: pd.DataFrame) -> pd.DataFrame:
    """
    Keep key review fields and create retrieval_text.
    """
    reviews = reviews[["parent_asin", "title", "text", "rating"]].copy()

    reviews["title"] = reviews["title"].fillna("").astype(str).str.strip()
    reviews["text"] = reviews["text"].fillna("").astype(str).str.strip()

    reviews["retrieval_text"] = (
        reviews["title"] + ". " + reviews["text"]
    ).str.replace(r"\s+", " ", regex=True).str.strip()

    reviews = reviews[reviews["retrieval_text"].str.len() > 0].reset_index(drop=True)
    return reviews


def preprocess_metadata(meta: pd.DataFrame) -> pd.DataFrame:
    """
    Keep key metadata fields, flatten list-like columns,
    and create retrieval_text.
    """
    meta = meta[["parent_asin", "title", "description", "features", "categories"]].copy()

    meta["title"] = meta["title"].fillna("").astype(str).str.strip()

    meta["description"] = meta["description"].apply(
        lambda x: " ".join(x) if isinstance(x, list) else ""
    )
    meta["features"] = meta["features"].apply(
        lambda x: " ".join(x) if isinstance(x, list) else ""
    )
    meta["categories"] = meta["categories"].apply(
        lambda x: " ".join(x) if isinstance(x, list) else ""
    )

    meta["retrieval_text"] = (
        meta["title"] + ". "
        + meta["description"].str.strip() + " "
        + meta["features"].str.strip() + " "
        + meta["categories"].str.strip()
    ).str.replace(r"\s+", " ", regex=True).str.strip()

    meta = meta[meta["retrieval_text"].str.len() > 0].reset_index(drop=True)
    return meta


def main():
    """Build and save the processed retrieval review and metadata datasets."""
    print("Building metadata lookup...")
    meta_lookup = build_metadata_lookup(META_PATH)
    print(f"Metadata records with non-empty title: {len(meta_lookup)}")

    print(f"Collecting reviews for at least {TARGET_PRODUCTS} unique products...")
    reviews_raw, meta_raw = collect_matched_reviews(
        REVIEW_PATH,
        meta_lookup,
        target_products=TARGET_PRODUCTS,
        max_reviews_per_product=MAX_REVIEWS_PER_PRODUCT,
    )

    print(f"Matched reviews shape: {reviews_raw.shape}")
    print(f"Matched metadata shape: {meta_raw.shape}")
    print(f"Unique products collected: {reviews_raw['parent_asin'].nunique()}")

    print("Preprocessing reviews...")
    reviews_processed = preprocess_reviews(reviews_raw)

    print("Preprocessing metadata...")
    meta_processed = preprocess_metadata(meta_raw)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    reviews_out = PROCESSED_DIR / "reviews_retrieval_processed.csv"
    meta_out = PROCESSED_DIR / "meta_retrieval_processed.csv"

    reviews_processed.to_csv(reviews_out, index=False)
    meta_processed.to_csv(meta_out, index=False)

    print(f"Saved reviews to: {reviews_out}")
    print(f"Saved metadata to: {meta_out}")
    print("Done.")


if __name__ == "__main__":
    main()
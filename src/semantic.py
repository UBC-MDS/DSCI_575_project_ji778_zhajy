import torch
import torch.nn as nn
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"

DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"

REVIEWS_PATH = PROCESSED_DIR / "reviews_retrieval_processed.csv"
META_PATH = PROCESSED_DIR / "meta_retrieval_processed.csv"
SEMANTIC_INDEX_PATH = PROCESSED_DIR / "semantic_faiss.index"
SEMANTIC_DOCS_PATH = PROCESSED_DIR / "semantic_documents.csv"


def load_documents(
    reviews_path: str | Path = REVIEWS_PATH,
    meta_path: str | Path = META_PATH,
) -> pd.DataFrame:
    """
    Load processed reviews and metadata, join them on parent_asin,
    and build review-level semantic documents with product title,
    review text, rating, and retrieval text.
    """
    reviews = pd.read_csv(reviews_path)
    meta = pd.read_csv(meta_path)

    reviews = reviews.copy()
    meta = meta.copy()

    meta_subset = meta[["parent_asin", "title"]].copy()
    meta_subset = meta_subset.rename(columns={"title": "product_title"})

    merged = reviews.merge(meta_subset, on="parent_asin", how="inner")
    merged["product_title"] = merged["product_title"].astype(str).str.strip()
    merged["title"] = merged["title"].fillna("").astype(str).str.strip()
    merged["text"] = merged["text"].fillna("").astype(str).str.strip()

    merged["retrieval_text"] = (
        merged["product_title"] + ". "
        + merged["title"] + ". "
        + merged["text"]
    ).str.replace(r"\s+", " ", regex=True).str.strip()

    merged = merged[merged["retrieval_text"].str.len() > 0].reset_index(drop=True)
    merged["source"] = "semantic"

    return merged


def get_embedding_model(model_name: str = MODEL_NAME) -> SentenceTransformer:
    """Load and return the sentence-transformers model."""
    return SentenceTransformer(model_name)


def encode_documents(
    documents: pd.DataFrame,
    model: SentenceTransformer,
    batch_size: int = 32,
) -> np.ndarray:
    """
    Create embeddings for document retrieval text.
    Returns a float32 numpy array for FAISS.
    """
    texts = documents["retrieval_text"].tolist()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings.astype("float32")


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Build a FAISS index from document embeddings.
    Because embeddings are normalized, inner product acts like cosine similarity.
    """
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index


def save_semantic_artifacts(
    index: faiss.Index,
    documents: pd.DataFrame,
    index_path: str | Path = SEMANTIC_INDEX_PATH,
    docs_path: str | Path = SEMANTIC_DOCS_PATH,
) -> None:
    """Save the FAISS index and aligned document table."""
    index_path = Path(index_path)
    docs_path = Path(docs_path)

    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    documents.to_csv(docs_path, index=False)


def build_and_save_semantic_index(
    reviews_path: str | Path = REVIEWS_PATH,
    meta_path: str | Path = META_PATH,
    index_path: str | Path = SEMANTIC_INDEX_PATH,
    docs_path: str | Path = SEMANTIC_DOCS_PATH,
    model_name: str = MODEL_NAME,
) -> tuple[faiss.Index, pd.DataFrame]:
    """
    Load processed reviews and metadata, build embeddings,
    construct a FAISS index, and save the semantic artifacts.
    """
    documents = load_documents(reviews_path=reviews_path, meta_path=meta_path)
    model = get_embedding_model(model_name=model_name)
    embeddings = encode_documents(documents, model=model)
    index = build_faiss_index(embeddings)
    save_semantic_artifacts(
        index=index,
        documents=documents,
        index_path=index_path,
        docs_path=docs_path,
    )
    return index, documents


def load_semantic_artifacts(
    index_path: str | Path = SEMANTIC_INDEX_PATH,
    docs_path: str | Path = SEMANTIC_DOCS_PATH,
) -> tuple[faiss.Index, pd.DataFrame]:
    """Load a saved FAISS index and aligned document table."""
    index = faiss.read_index(str(index_path))
    documents = pd.read_csv(docs_path)
    return index, documents


def encode_query(
    query: str,
    model: SentenceTransformer,
) -> np.ndarray:
    """
    Encode a query using the same model and normalization as the documents.
    This keeps document and query embeddings consistent.
    """
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return query_embedding.astype("float32")


def semantic_search(
    query: str,
    index: faiss.Index,
    documents: pd.DataFrame,
    model: SentenceTransformer,
    top_k: int = 5,
) -> pd.DataFrame:
    """
    Search the FAISS index with a query and return ranked results with scores.
    Scores are cosine-similarity-like because normalized embeddings are used
    with inner-product search.
    """
    query_embedding = encode_query(query, model=model)
    scores, indices = index.search(query_embedding, top_k)

    result_indices = indices[0]
    result_scores = scores[0]

    results = documents.iloc[result_indices].copy().reset_index(drop=True)
    results["score"] = result_scores
    results["rank"] = range(1, len(results) + 1)

    return results[["rank", "score"] + [col for col in results.columns if col not in {"rank", "score"}]]


if __name__ == "__main__":
    build_and_save_semantic_index()
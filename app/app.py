import torch
import torch.nn as nn
import html
import re
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.bm25 import BM25Retriever
from src.prompts import SYSTEM_PROMPT_1
from src.rag_pipeline import run_pipeline_with_tool
from src.semantic import get_embedding_model, load_semantic_artifacts, semantic_search

DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"

BM25_INDEX_PATH = PROCESSED_DIR / "bm25_index.pkl"
BM25_CORPUS_PATH = PROCESSED_DIR / "corpus_data.pkl"
SEMANTIC_INDEX_PATH = PROCESSED_DIR / "semantic_faiss.index"
SEMANTIC_DOCS_PATH = PROCESSED_DIR / "semantic_documents.csv"

st.set_page_config(page_title="Movie & TV Retrieval App", page_icon="🎬", layout="wide")


@st.cache_resource
def load_bm25() -> BM25Retriever:
    """Load and cache the BM25 retriever."""
    retriever = BM25Retriever(
        index_path=str(BM25_INDEX_PATH),
        corpus_path=str(BM25_CORPUS_PATH),
    )
    retriever.load_index()
    return retriever


@st.cache_resource
def load_semantic():
    """Load and cache the semantic model, index, and documents."""
    model = get_embedding_model()
    index, documents = load_semantic_artifacts(
        index_path=str(SEMANTIC_INDEX_PATH),
        docs_path=str(SEMANTIC_DOCS_PATH),
    )
    return model, index, documents


def format_rating(rating) -> str:
    """Format numeric ratings as stars plus the numeric value."""
    if pd.isna(rating):
        return "N/A"
    try:
        rating = float(rating)
        full_stars = int(round(rating))
        return f"{'★' * full_stars} ({rating:.1f})"
    except (ValueError, TypeError):
        return str(rating)


def clean_review_text(text: str) -> str:
    """Clean HTML tags and whitespace from review text."""
    if not isinstance(text, str):
        return "N/A"

    text = html.unescape(text)
    text = re.sub(r"<br\s*/?>", "\n", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def truncate_text(text: str, max_chars: int = 200) -> str:
    """Return a shortened preview of review text."""
    text = clean_review_text(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def display_result_card(
    title: str,
    review_text: str,
    rating,
    score=None,
    source: str | None = None,
) -> None:
    """Display one retrieval result card in the app."""
    cleaned_review = clean_review_text(review_text)

    st.markdown(f"### {title if title else 'Untitled'}")
    st.write(f"**Review preview:** {truncate_text(cleaned_review)}")

    if isinstance(cleaned_review, str) and cleaned_review.strip():
        with st.expander("Show full review"):
            st.write(cleaned_review)

    st.write(f"**Rating:** {format_rating(rating)}")

    if score is not None:
        if isinstance(score, (int, float)):
            st.write(f"**Retrieval score:** {score:.4f}")
        else:
            st.write(f"**Retrieval score:** {score}")

    if source:
        st.caption(f"Source: {source}")

    st.markdown("---")


def run_bm25_search(query: str, top_k: int = 3):
    """Run BM25 search for a query."""
    bm25 = load_bm25()
    return bm25.search(query, top_n=top_k)


def run_semantic_search(query: str, top_k: int = 3) -> pd.DataFrame:
    """Run semantic search for a query."""
    model, index, documents = load_semantic()
    return semantic_search(
        query=query,
        index=index,
        documents=documents,
        model=model,
        top_k=top_k,
    )


st.title("Movie & TV Retrieval App")
st.write("Search reviews and product information, or use Hybrid RAG to generate grounded answers.")

search_tab, rag_tab = st.tabs(["Search", "RAG"])

with search_tab:
    st.subheader("Search Only")

    with st.form("search_form"):
        search_mode = st.radio("Search mode", ["BM25", "Semantic"], horizontal=True)
        query = st.text_input("Enter your search query")
        submitted = st.form_submit_button("Search")

    if submitted:
        if not query.strip():
            st.warning("Please enter a query.")
        else:
            with st.spinner("Searching..."):
                if search_mode == "BM25":
                    results = run_bm25_search(query, top_k=3)

                    if not results:
                        st.info("No results found.")
                    else:
                        for doc, score in results:
                            title = doc.get("product_title") or doc.get("title") or "Untitled"
                            review_text = doc.get("review_text", "")
                            rating = doc.get("rating", "N/A")

                            display_result_card(
                                title=title,
                                review_text=review_text,
                                rating=rating,
                                score=score,
                                source="BM25",
                            )

                elif search_mode == "Semantic":
                    results = run_semantic_search(query, top_k=3)

                    if results.empty:
                        st.info("No results found.")
                    else:
                        for _, row in results.iterrows():
                            title = row.get("product_title", row.get("title", "Untitled"))
                            review_text = row.get("text", "")
                            if not isinstance(review_text, str) or not review_text.strip():
                                review_text = row.get("retrieval_text", "")
                            rating = row.get("rating", "N/A")
                            score = row.get("score", "N/A")
                            source = row.get("source", "Semantic")

                            display_result_card(
                                title=title,
                                review_text=review_text,
                                rating=rating,
                                score=score,
                                source=source,
                            )

with rag_tab:
    st.subheader("Hybrid RAG")
    rag_query = st.text_input("Enter your RAG query", key="rag_query")

    if st.button("Generate Answer", key="rag_button"):
        if not rag_query.strip():
            st.warning("Please enter a query.")
        else:
            with st.spinner("Generating answer..."):
                answer, docs, mode = run_pipeline_with_tool(
                    rag_query,
                    system_prompt=SYSTEM_PROMPT_1,
                )

                st.markdown("## RAG Answer")
                st.write(answer)
                st.caption(mode)

                st.markdown("## Retrieved Supporting Documents")
                if docs is None or docs.empty:
                    st.info("No supporting documents found.")
                else:
                    for _, row in docs.iterrows():
                        title = row.get("product_title", "Untitled")
                        review_text = row.get("text", "")
                        rating = row.get("rating", "N/A")
                        score = row.get("score", None)
                        source = row.get("source", "Hybrid")

                        display_result_card(
                            title=title,
                            review_text=review_text,
                            rating=rating,
                            score=score,
                            source=source,
                        )
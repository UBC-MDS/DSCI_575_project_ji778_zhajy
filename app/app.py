import torch
import torch.nn as nn
import streamlit as st
import pandas as pd
import sys
import re
import html
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.bm25 import BM25Retriever
from src.semantic import get_embedding_model, load_semantic_artifacts, semantic_search
from src.rag_pipeline import run_hybrid_rag_pipeline

st.set_page_config(page_title="Movie & TV Retrieval App", page_icon="🎬", layout="wide")


@st.cache_resource
def load_bm25():
    retriever = BM25Retriever(
        index_path="data/processed/bm25_index.pkl",
        corpus_path="data/processed/corpus_data.pkl",
    )
    retriever.load_index()
    return retriever


@st.cache_resource
def load_semantic():
    model = get_embedding_model()
    index, documents = load_semantic_artifacts(
        index_path="data/processed/semantic_faiss.index",
        docs_path="data/processed/semantic_documents.csv",
    )
    return model, index, documents


def format_rating(rating):
    if pd.isna(rating):
        return "N/A"
    try:
        rating = float(rating)
        full_stars = int(round(rating))
        return f"{'★' * full_stars} ({rating:.1f})"
    except (ValueError, TypeError):
        return str(rating)


def clean_review_text(text):
    if not isinstance(text, str):
        return "N/A"

    text = html.unescape(text) 
    text = re.sub(r"<br\s*/?>", "\n", text)
    text = re.sub(r"<[^>]+>", "", text) 
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def truncate_text(text, max_chars=200):
    text = clean_review_text(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def display_result_card(title, review_text, rating, score=None, source=None):
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


def run_bm25_search(query, top_k=3):
    bm25 = load_bm25()
    results = bm25.search(query, top_n=top_k)
    return results


def run_semantic_search(query, top_k=3):
    model, index, documents = load_semantic()
    results = semantic_search(
        query=query,
        index=index,
        documents=documents,
        model=model,
        top_k=top_k,
    )
    return results


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
                answer, docs = run_hybrid_rag_pipeline(rag_query)

                st.markdown("## RAG Answer")
                st.write(answer)

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
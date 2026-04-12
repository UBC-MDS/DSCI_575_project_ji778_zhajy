import torch
import torch.nn as nn
import sys
import os
import pandas as pd

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(base_dir)

from src.bm25 import BM25Retriever
from src.semantic import load_semantic_artifacts, get_embedding_model, semantic_search

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "" 
    
    bm25_path = os.path.join(base_dir, 'data/processed/bm25_index.pkl')
    corpus_path = os.path.join(base_dir, 'data/processed/corpus_data.pkl')
    
    print("Loading BM25...")
    bm25 = BM25Retriever(index_path=bm25_path, corpus_path=corpus_path)
    bm25.load_index()

    faiss_path = os.path.join(base_dir, 'data/processed/faiss_index.bin')
    docs_path = os.path.join(base_dir, 'data/processed/combined_documents.csv')
    
    print("Loading Semantic Search...")
    faiss_idx, semantic_docs = load_semantic_artifacts(
        index_path=faiss_path, 
        docs_path=docs_path
    )
    encoder_model = get_embedding_model()

    queries = [
        "Glee season 4 musical comedy",
        "Taylor Cole Jack Turner chalet wedding",
        "Texas sheriff border corruption movie",
        "funny tv series about high school singing competitions",
        "romantic movie about a couple trying to plan a small getaway marriage",
        "police officer thriller investigating murders and dark secrets",
        "comedy about a guy going to wild parties in Las Vegas to find a girlfriend",
        "independent film about mental illness with an ending that lacks closure",
        "family friendly shows that young kids will absolutely love to watch",
        "highly rated show that amazon needs to buy and bring back"
    ]

    print("\n" + "="*50)
    print("QUALITATIVE EVALUATION RESULTS")
    print("="*50)

    for q in queries:
        print(f"\nQUERY: '{q}'")
        print("-" * 30)
        
        print("BM25 Results:")
        bm25_results = bm25.search(q, top_n=3)
        if not bm25_results:
            print("  - No results found.")
        for doc, score in bm25_results:
            print(f"  - [{score:.2f}] {doc.get('product_title', 'Unknown')}")

        print("\nSemantic Results:")
        sem_res = semantic_search(q, faiss_idx, semantic_docs, encoder_model, top_k=3)
        for row in sem_res.itertuples():
            title = getattr(row, 'product_title', getattr(row, 'title', 'Unknown'))
            score = getattr(row, 'score', 0.0)
            print(f"  - [{score:.4f}] {title}")

if __name__ == "__main__":
    main()
import pickle
import os
from rank_bm25 import BM25Okapi
from src.utils import tokenize_text

class BM25Retriever:
    def __init__(self, index_path="data/processed/bm25_index.pkl", corpus_path="data/processed/corpus_data.pkl"):
        self.index_path = index_path
        self.corpus_path = corpus_path
        self.bm25_model = None
        self.tokenized_corpus = None
        self.corpus_metadata = None 

    def build_and_save_index(self, corpus_metadata):
        """
        Takes a list of dictionaries (containing retrieval_text, product_title, review_text, rating),
        tokenizes the retrieval_text, and saves the index.
        """
        print("Tokenizing corpus...")
        self.corpus_metadata = corpus_metadata
        
        self.tokenized_corpus = [tokenize_text(doc['retrieval_text']) for doc in corpus_metadata]
        
        print("Building BM25 Index...")
        self.bm25_model = BM25Okapi(self.tokenized_corpus)
        
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        with open(self.index_path, 'wb') as f:
            pickle.dump(self.bm25_model, f)
            
        with open(self.corpus_path, 'wb') as f:
            pickle.dump(self.corpus_metadata, f)
            
        print(f"Index successfully saved to {self.index_path}")

    def load_index(self):
        if not os.path.exists(self.index_path) or not os.path.exists(self.corpus_path):
            raise FileNotFoundError("Index files not found. Run build_and_save_index first.")
            
        with open(self.index_path, 'rb') as f:
            self.bm25_model = pickle.load(f)
            
        with open(self.corpus_path, 'rb') as f:
            self.corpus_metadata = pickle.load(f)

    def search(self, query, top_n=5):
        """Returns the top N matching dictionaries AND their BM25 scores."""
        if self.bm25_model is None:
            raise ValueError("BM25 model is not loaded.")
            
        tokenized_query = tokenize_text(query)
        
        scores = self.bm25_model.get_scores(tokenized_query)
        
        scored_results = list(zip(self.corpus_metadata, scores))
        scored_results.sort(key=lambda x: x[1], reverse=True)
        
        return scored_results[:top_n]
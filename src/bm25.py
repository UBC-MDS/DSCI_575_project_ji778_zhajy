import pickle
from pathlib import Path

from rank_bm25 import BM25Okapi

from src.utils import tokenize_text

DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"

DEFAULT_INDEX_PATH = PROCESSED_DIR / "bm25_index.pkl"
DEFAULT_CORPUS_PATH = PROCESSED_DIR / "corpus_data.pkl"


class BM25Retriever:
    """BM25 retriever for keyword-based search over review documents."""

    def __init__(
        self,
        index_path: str | Path = DEFAULT_INDEX_PATH,
        corpus_path: str | Path = DEFAULT_CORPUS_PATH,
    ) -> None:
        """Initialize the BM25 retriever with artifact paths."""
        self.index_path = Path(index_path)
        self.corpus_path = Path(corpus_path)
        self.bm25_model = None
        self.tokenized_corpus = None
        self.corpus_metadata = None

    def build_and_save_index(self, corpus_metadata: list[dict]) -> None:
        """
        Tokenize the retrieval text, build the BM25 index,
        and save the index and corpus metadata.
        """
        print("Tokenizing corpus...")
        self.corpus_metadata = corpus_metadata

        self.tokenized_corpus = [
            tokenize_text(doc["retrieval_text"]) for doc in corpus_metadata
        ]

        print("Building BM25 Index...")
        self.bm25_model = BM25Okapi(self.tokenized_corpus)

        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.index_path, "wb") as f:
            pickle.dump(self.bm25_model, f)

        with open(self.corpus_path, "wb") as f:
            pickle.dump(self.corpus_metadata, f)

        print(f"Index successfully saved to {self.index_path}")

    def load_index(self) -> None:
        """Load the saved BM25 index and corpus metadata from disk."""
        if not self.index_path.exists() or not self.corpus_path.exists():
            raise FileNotFoundError("Index files not found. Run build_and_save_index first.")

        with open(self.index_path, "rb") as f:
            self.bm25_model = pickle.load(f)

        with open(self.corpus_path, "rb") as f:
            self.corpus_metadata = pickle.load(f)

    def search(self, query: str, top_n: int = 5) -> list[tuple[dict, float]]:
        """Return the top matching documents and their BM25 scores."""
        if self.bm25_model is None:
            raise ValueError("BM25 model is not loaded.")

        tokenized_query = tokenize_text(query)
        scores = self.bm25_model.get_scores(tokenized_query)

        scored_results = list(zip(self.corpus_metadata, scores))
        scored_results.sort(key=lambda x: x[1], reverse=True)

        return scored_results[:top_n]
# DSCI_575_project_ji778_zhajy

# Movie & TV Retrieval System

## Project overview

This project builds a small retrieval system for the Amazon Movies and TV dataset. The goal is to let a user search product-related content using two retrieval methods:

- BM25 search for keyword-based retrieval
- Semantic search for embedding-based retrieval

A simple Streamlit web app is included so users can enter a query, choose a search mode, and view the top results.

For each result, the app displays:
- product title
- truncated review text
- rating
- retrieval score

## Environment setup

Clone the repository and create the environment with:

```bash
git clone https://github.com/UBC-MDS/DSCI_575_project_ji778_zhajy.git
cd DSCI_575_project_ji778_zhajy.git
conda env create -f environment.yml
conda activate dsci-575-project
```
## Dataset

This project uses the [Amazon Movies and TV dataset](https://amazon-reviews-2023.github.io/).

Two raw files are required:

- `Movies_and_TV.jsonl.gz`
- `meta_Movies_and_TV.jsonl.gz`

Both files should be placed in data/raw/.

### Subset used in this project

For reproducibility, this project currently uses a matched 200-row subset. A subset of 200 review records is used, and the metadata records are selected by matching `parent_asin` values rather than being sampled independently. This keeps the review data and metadata aligned and helps avoid mismatched review and product records in the retrieval workflow. 

## Data processing

This project uses both review data and metadata to prepare documents for retrieval. For the review data, the main fields used are `parent_asin`, `title`, `text`, and `rating`. For the metadata, the main fields used are `parent_asin`, `title`, `description`, `features`, and `categories`.

The preprocessing is designed to keep the text useful for retrieval while preserving the link between reviews and products. In the review data, only the selected fields are kept, missing text fields are filled with empty strings, and the review `title` and `text` are combined into a single `retrieval_text` field. In the metadata, only the selected fields are kept, and list-like fields such as `description`, `features`, and `categories` are flattened into plain text before being combined into `retrieval_text`.

The field `parent_asin` is preserved throughout the workflow because it is used to match review records with metadata records. This matching step is important because product titles come from the metadata file, while review text and ratings come from the review file.

The notebook `notebooks/milestone1_exploration.ipynb` is used for exploratory analysis on the matched 200-row subset. It also saves the processed sample files to:

- `data/processed/reviews_sample_processed.csv`
- `data/processed/meta_sample_processed.csv`

## Retrieval workflows

### BM25 retrieval

The BM25 workflow is implemented in `src/bm25.py`. The BM25 tokenizer lowercases text, removes punctuation, removes stopwords, and tokenizes the text before indexing. The BM25 retriever saves the indexed outputs to:

- `data/processed/bm25_index.pkl`
- `data/processed/corpus_data.pkl`

Given a query, the BM25 retriever tokenizes the query, computes BM25 scores, and returns ranked results with retrieval scores.

### Semantic retrieval

The semantic workflow is implemented in `src/semantic.py`. The semantic pipeline loads the processed review and metadata files, aligns review records with metadata using `parent_asin`, and builds semantic documents using product title, review title, and review text. It then generates embeddings using the `sentence-transformers` model `all-MiniLM-L6-v2`, builds the semantic index, and saves the semantic artifacts to:

- `data/processed/semantic_documents.csv`
- `data/processed/semantic_faiss.index`

## Run the app locally

The web app is implemented in `app/app.py`.

To launch the app locally from the project root, run:

```bash
streamlit run app/app.py
```

# Final Discussion

## Step 1: Improve Your Workflow

### Dataset Scaling
- Number of products used: 10,000
- Changes to sampling strategy (if any): The sampling strategy was updated to scale the dataset to 10,000 unique products while maximizing diversity. The script streams the raw files and enforces a strict limit of 1 review per product (`MAX_REVIEWS_PER_PRODUCT = 1`). It also ensures data alignment and quality by verifying every collected review has a corresponding metadata record with a valid, non-empty product title, halting the collection as soon as the 10,000 unique product threshold is reached.

### LLM Experiment
- Models compared (name, family, size): Groq llama-3.1-8b-instant (Llama 3 family, 8B parameters) vs. google/flan-t5-large (T5 family, ~780M parameters).
- Results and discussions: 
    - Prompt used (copy it here): You are a Movie and TV recommendation expert. Answer the user's query using ONLY the provided reviews and metadata below. If the information is not present in the context, clearly state that you cannot find the answer. To ensure accuracy, always cite the Amazon ASIN for every title you discuss.
    - Results: 
        - **Query 1 (Glee season 4):** Groq provided a detailed answer discussing the musical/comedic aspects and cited the ASIN `B009LDD5E0`. FLAN-T5 only returned the title `Glee: Season 4`.
        - **Query 2 (Taylor Cole chalet wedding):** Groq accurately stated the context lacked evidence for this specific query but noted limited information on Aidan Turner. FLAN-T5 simply returned `Cannot find the answer`.
        - **Query 3 (Texas sheriff):** Groq attempted to infer a match (*The Alamo*) but noted the context did not strongly support it. FLAN-T5 returned a short extraction `[1] Texas Killing Fields` without explanation.
        - **Query 4 (Vegas wild parties):** Groq identified *Can't Hardly Wait* as the closest match while correctly noting it was not set in Las Vegas. FLAN-T5 only returned the title.
        - **Query 5 (Mental illness independent film):** Groq identified *Mental* as the closest match and explicitly explained that the context did not mention an ending lacking closure. FLAN-T5 mostly copied the raw review text without interpretation.
        - **Key Observations:** The Groq model generally produced more complete and useful answers. It was better at explaining uncertainty, connecting the query to the retrieved reviews, and providing a natural final response. FLAN-T5 often returned only a title or a very short extraction. Furthermore, both models were ultimately limited by retrieval quality; when the retrieved context did not clearly match the query, even the stronger Groq model struggled.
- Which model you chose and why: We chose to keep **Groq llama-3.1-8b-instant** as the default model. It consistently produced clearer, more complete, and more natural answers than google/flan-t5-large, so the pipeline was not updated to switch to the second model.

## Step 2: Additional Feature (Option 2: Tool Integration)

### What You Implemented

- Description of the feature: We integrated an external Web Search tool (`DuckDuckGoSearchRun` via LangChain) into our pipeline. We designed a conditional routing system: if the LLM determines that the retrieved Amazon review context is insufficient to answer the user's query (by detecting failure phrases like "cannot find"), the pipeline automatically triggers DuckDuckGo to search the live web for the answer.
- Key results or examples: 
    Here are 3 example queries demonstrating the routing system in action:

    **Query 1: "Who won the Best Picture Oscar in 2024?"**
    * **Routing:** Web Search Used
    * **Result:** The tool successfully triggered and retrieved live data, correctly identifying Emma Thomas, Charles Roven, and Christopher Nolan for 'Oppenheimer'.

    **Query 2: "What is the plot of the movie Dune: Part Two?"**
    * **Routing:** Web Search Used
    * **Result:** The tool successfully fetched external plot details focusing on Paul Atreides' fate, providing an answer that was entirely absent from our Amazon dataset.

    **Query 3: "Glee season 4 musical comedy cast"**
    * **Routing:** Hybrid RAG Used (Tool Bypassed)
    * **Result:** The LLM stated the cast was "not present in the given context." Because the model phrased its failure differently than our hardcoded trigger phrases, the web tool was not activated. 

    **Did it improve the results?**
    Yes, it significantly improved the robustness of the system. For out-of-domain or highly recent queries (like the 2024 Oscars), the system now seamlessly falls back to the internet to provide accurate answers instead of hitting a dead end. Query 3 also provided a valuable insight: to improve the workflow further, we should use an LLM-based routing agent to decide when to use the tool, rather than relying on hardcoded string matching.

## Step 3: Improve Documentation and Code Quality

### Documentation Update
- Summary of README improvements: Updated the README with clear setup instructions, usage examples, and architecture diagrams.

### Code Quality Changes
- Summary of cleanups: Removed hardcoded paths using `pathlib`, added docstrings to all functions, updated the `requirements.txt` environment file, and updated the `.gitignore` to exclude large data artifacts.

## Step 4: Cloud Deployment Plan

### 1. Data Storage
- **Raw Data:** Store in an S3 Bucket (AWS) for durable, low-cost object storage.
- **Processed Data:** Store as Parquet files in S3 or a managed database like AWS RDS if relational queries are needed.
- **Vector Index:** Host using a managed vector database such as Pinecone or Milvus to allow for high-concurrency retrieval.
- **BM25 Index:** Store as a serialized artifact in S3 or utilize Elasticsearch (OpenSearch) for production-grade keyword searching.

### 2. Compute
- **App Execution:** The app would run on AWS App Runner or EC2 instances behind an Elastic Load Balancer.
- **Concurrency:** Use a Load Balancer to distribute traffic across multiple containerized instances (Docker) to handle multiple users simultaneously.
- **LLM Inference:** Use a managed API (like Groq or AWS Bedrock) for scalability and reduced maintenance overhead.

### 3. Streaming/Updates
- **New Products:** Implement a CI/CD pipeline that triggers a "Data Refresh" job when new data lands in the S3 bucket.
- **Pipeline Updates:** Use AWS Lambda to trigger incremental updates to the Vector and BM25 indices whenever new products are added, ensuring the app stays current without full redeployments.
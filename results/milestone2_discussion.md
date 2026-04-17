# Milestone 2

## Discussion for building LLM pipeline

### Model choice and rationale

For the LLM pipeline, this project uses Groq with the model `llama-3.1-8b-instant`. This model was chosen because it avoids the need to download and run a large model locally, which makes the workflow much more practical on a laptop. It also allows the generation component to be tested quickly before integrating retrieval and prompt engineering in the later RAG steps.

This choice provides a good balance between response quality and ease of use. Since the goal of this milestone is to build a working RAG pipeline for review-based question answering, the main requirement is a model that can generate clear and relevant answers from provided context. Using Groq keeps the setup lightweight and reproducible while still giving strong enough performance for prototyping and demos.

### Prompt Engineering Experiments
To optimize our RAG pipeline, we tested three different System Prompt variants using the query: "romantic comedy movie about weddings". We evaluated the outputs based on Accuracy (grounding), Completeness, and Fluency.

#### Variant 1 (The Precise Critic): 
This prompt provided the best balance of Accuracy and readability. It strictly adhered to the RAG context, directly quoting the short review snippets (e.g., noting that Christmas Wedding Planner was described as "nice and clean"). It correctly cited ASINs and ratings without injecting outside knowledge.

#### Variant 2 (The Community Guide): 
This prompt scored highest in Fluency and Completeness (listing 5 movies instead of 3), sounding very natural. However, it failed significantly on Accuracy/Grounding. The conversational persona caused the LLM to hallucinate full plot summaries that were not present in the provided Amazon review text, relying on its internal pre-trained knowledge instead.

#### Variant 3 (The Fact-Checker): 
This prompt was highly accurate but lacked Fluency. It acted as a strict data extractor, merely listing the Title, ASIN, and Rating in a bulleted list. While safe from hallucinations, it was too robotic for a consumer-facing movie recommendation app.

#### Final Decision: 
We selected Variant 1 (The Precise Critic) as our default prompt. It prevents the model from hallucinating plot details while still providing a readable, well-formatted recommendation based strictly on the retrieved Amazon reviews.

## Qualitative Evaluation (Hybrid RAG)

We evaluated 5 specific queries originally formulated in Milestone 1 using our Hybrid RAG pipeline and the "Precise Critic" system prompt.

| Query | Accuracy (Y/N) | Completeness (Y/N) | Fluency (Y/N) |
| :--- | :---: | :---: | :---: |
| 1. Glee season 4 musical comedy | Y | N | Y |
| 2. Taylor Cole Jack Turner chalet wedding | Y | N | Y |
| 3. Texas sheriff border corruption movie | Y | Y | Y |
| 4. comedy about a guy going to wild parties in Las Vegas to find a girlfriend | Y | N | Y |
| 5. independent film about mental illness with an ending that lacks closure | Y | Y | Y |

### Summary and Reflections

**Key Observations & Overall Performance:** The Hybrid RAG workflow showed excellent **Accuracy** and **Fluency**. Our "Precise Critic" system prompt successfully prevented the model from hallucinating. For example, in Query 1 and 2, when the retriever failed to find "Glee Season 4" or the "Chalet Wedding" movie, the LLM honestly admitted the information was missing rather than making up a fake Amazon review. However, the overall **Completeness** suffered heavily due to the retrieval step.

**Limitations of the Hybrid RAG Workflow:**
The primary limitation is the hard `Top-K = 5` cutoff during retrieval. For highly specific queries (like Query 2), the actual target movie likely ranked just outside the top 5, leaving the LLM blind to it. Second, the Reciprocal Rank Fusion (RRF) scoring might be diluting highly specific semantic matches with irrelevant BM25 keyword matches, causing the correct document to drop in rank.

**Suggestions for Improvement:**
To improve the workflow, we should increase the `Top-K` value passed from the retriever to the context builder (e.g., retrieving 10 or 15 documents instead of 5) to give the LLM a wider net of information. Additionally, implementing a Cross-Encoder Reranker after the Hybrid search would help push the most semantically relevant documents to the top before the cutoff is applied.
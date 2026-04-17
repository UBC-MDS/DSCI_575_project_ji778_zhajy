# Milestone 2

## Discussion for building LLM pipeline

### Model choice and rationale

For the LLM pipeline, this project uses Groq with the model `llama-3.1-8b-instant`. This model was chosen because it avoids the need to download and run a large model locally, which makes the workflow much more practical on a laptop. It also allows the generation component to be tested quickly before integrating retrieval and prompt engineering in the later RAG steps.

This choice provides a good balance between response quality and ease of use. Since the goal of this milestone is to build a working RAG pipeline for review-based question answering, the main requirement is a model that can generate clear and relevant answers from provided context. Using Groq keeps the setup lightweight and reproducible while still giving strong enough performance for prototyping and demos.

### Prompt Engineering & Qualitative Evaluation
To optimize our RAG pipeline, we tested three different System Prompt variants using the query: "romantic comedy movie about weddings". We evaluated the outputs based on Accuracy (grounding), Completeness, and Fluency.

#### Variant 1 (The Precise Critic): 
This prompt provided the best balance of Accuracy and readability. It strictly adhered to the RAG context, directly quoting the short review snippets (e.g., noting that Christmas Wedding Planner was described as "nice and clean"). It correctly cited ASINs and ratings without injecting outside knowledge.

#### Variant 2 (The Community Guide): 
This prompt scored highest in Fluency and Completeness (listing 5 movies instead of 3), sounding very natural. However, it failed significantly on Accuracy/Grounding. The conversational persona caused the LLM to hallucinate full plot summaries that were not present in the provided Amazon review text, relying on its internal pre-trained knowledge instead.

#### Variant 3 (The Fact-Checker): 
This prompt was highly accurate but lacked Fluency. It acted as a strict data extractor, merely listing the Title, ASIN, and Rating in a bulleted list. While safe from hallucinations, it was too robotic for a consumer-facing movie recommendation app.

#### Final Decision: 
We selected Variant 1 (The Precise Critic) as our default prompt. It prevents the model from hallucinating plot details while still providing a readable, well-formatted recommendation based strictly on the retrieved Amazon reviews.
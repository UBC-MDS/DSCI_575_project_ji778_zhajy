# Milestone 2

## Discussion for building LLM pipeline

### Model choice and rationale

For the LLM pipeline, this project uses Groq with the model `llama-3.1-8b-instant`. This model was chosen because it avoids the need to download and run a large model locally, which makes the workflow much more practical on a laptop. It also allows the generation component to be tested quickly before integrating retrieval and prompt engineering in the later RAG steps.

This choice provides a good balance between response quality and ease of use. Since the goal of this milestone is to build a working RAG pipeline for review-based question answering, the main requirement is a model that can generate clear and relevant answers from provided context. Using Groq keeps the setup lightweight and reproducible while still giving strong enough performance for prototyping and demos.

## LLM Comparison

### Models used

The two models compared in this experiment were:

-   Groq llama-3.1-8b-instant
-   google/flan-t5-large

The purpose of this comparison was to see how model quality affects the final RAG response when the retrieved context and prompt are kept fixed.

### Prompt

The same prompt was used for both models in all five queries:

You are a Movie and TV recommendation expert. Answer the user's query using ONLY the provided reviews and metadata below. If the information is not present in the context, clearly state that you cannot find the answer. To ensure accuracy, always cite the Amazon ASIN for every title you discuss.

### Outputs

#### Query 1: Glee season 4 musical comedy

-   Groq output: The Groq model gave a detailed answer discussing the musical and comedic aspects of *Glee: Season 4*, explained the reviewer’s mixed reaction, and cited the ASIN `B009LDD5E0`.
-   FLAN-T5 output: The Hugging Face model only returned `Glee: Season 4`.

#### Query 2: Taylor Cole Jack Turner chalet wedding

-   Groq output: The Groq model stated that the retrieved context did not contain enough evidence for Taylor Cole or a chalet wedding, and correctly said it could only find limited information related to Aidan Turner.
-   FLAN-T5 output: The Hugging Face model returned `Cannot find the answer`.

#### Query 3: Texas sheriff border corruption movie

-   Groq output: The Groq model tried to infer a possible match and mentioned *The Alamo*, but the answer was not very accurate because the retrieved context did not strongly support the query.
-   FLAN-T5 output: The Hugging Face model returned `[1] Texas Killing Fields`, which was short and not well explained.

#### Query 4: comedy about a guy going to wild parties in Las Vegas to find a girlfriend

-   Groq output: The Groq model gave a more nuanced answer, identifying *Can't Hardly Wait* as the closest match while also noting that it was not actually set in Las Vegas.
-   FLAN-T5 output: The Hugging Face model only returned `Can't Hardly Wait`.

#### Query 5: independent film about mental illness with an ending that lacks closure

-   Groq output: The Groq model identified *Mental* as the closest available match, explained that the available context did not mention the ending lacking closure, and mentioned other related titles cautiously.
-   FLAN-T5 output: The Hugging Face model mostly copied the retrieved review text for *Mental* without much interpretation.

### Key observations

The Groq model generally produced more complete and useful answers across the five queries. It was better at explaining uncertainty, connecting the query to the retrieved reviews, and giving a more natural final response. By contrast, `google/flan-t5-large` often returned only a title or a very short extraction from the context, which made its answers less informative.

A second observation is that both models were still limited by retrieval quality. When the retrieved context did not clearly match the query, even the stronger Groq model sometimes made weak inferences or could only partially answer the question. This suggests that model quality matters, but retrieval quality still remains a major bottleneck in the overall RAG workflow.

### Final model choice

Based on this comparison, Groq llama-3.1-8b-instant remains the better default model for the pipeline. It consistently produced clearer, more complete, and more natural answers than google/flan-t5-large, so the pipeline was not updated to switch to the second model.

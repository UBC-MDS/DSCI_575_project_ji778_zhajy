"""
Prompt templates for Movie & TV RAG.
Refined for Milestone 2 Qualitative Evaluation (Accuracy, Completeness, Fluency).
"""

def build_context(docs_df):
    """
    Formats retrieved documents into a structured block.
    Matches the columns in meta_retrieval_processed and reviews_retrieval_processed.
    """
    context_parts = []
    for i, (_, row) in enumerate(docs_df.iterrows(), 1):
        asin = row.get('parent_asin', 'N/A')
        title = row.get('product_title', 'Unknown Title')
        rating = row.get('rating', 'N/A')
        review_body = row.get('text', 'No review text available.')

        block = (
            f"--- Entry {i} ---\n"
            f"Title: {title}\n"
            f"ASIN: {asin}\n"
            f"Rating: {rating}/5 stars\n"
            f"Review: {review_body}\n"
        )
        context_parts.append(block)
    
    return "\n".join(context_parts)

SYSTEM_PROMPT_1 = (
    "You are a Movie and TV recommendation expert. "
    "Answer the user's query using ONLY the provided reviews and metadata below. "
    "If the information is not present in the context, clearly state that you cannot find the answer. "
    "To ensure accuracy, always cite the Amazon ASIN for every title you discuss."
)

SYSTEM_PROMPT_2 = (
    "You are a knowledgeable guide for the Movie & TV community. "
    "Using the provided viewer reviews, help the user find exactly what they are looking for. "
    "Prioritize exact title matches for shows like 'Glee' or specific actors mentioned in the context. "
    "Always reference the ASIN and the star rating so the user knows which version you recommend."
)

SYSTEM_PROMPT_3 = (
    "You are a precise data extractor for Movies and TV shows. "
    "Your task is to answer questions concisely using the provided context blocks. "
    "Do not use outside knowledge. If the context mentions multiple products, "
    "differentiate them by their unique ASIN and product title."
)

def build_prompt(query, context, system_prompt=SYSTEM_PROMPT_1):
    """
    Combines system prompt, context, and query for the LLM [cite: 120-125].
    """
    return (
        f"{system_prompt}\n\n"
        f"CONTEXT FROM AMAZON DATASET:\n{context}\n\n"
        f"USER QUERY: {query}\n\n"
        f"FINAL RESPONSE:"
    )
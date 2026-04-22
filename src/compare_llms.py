import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.hybrid import hybrid_retriever
from src.prompts import SYSTEM_PROMPT_1, build_prompt
from src.rag_pipeline import build_context

GROQ_MODEL_NAME = "llama-3.1-8b-instant"
HF_MODEL_NAME = "google/flan-t5-large"

QUERIES = [
    "Glee season 4 musical comedy",
    "Taylor Cole Jack Turner chalet wedding",
    "Texas sheriff border corruption movie",
    "comedy about a guy going to wild parties in Las Vegas to find a girlfriend",
    "independent film about mental illness with an ending that lacks closure",
]


def load_groq_model() -> ChatGroq:
    """Load and return the Groq chat model used for comparison."""
    load_dotenv()
    return ChatGroq(
        model=GROQ_MODEL_NAME,
        api_key=os.getenv("GROQ_API_KEY"),
    )


def load_hf_pipeline():
    """Load and return the Hugging Face text2text generation pipeline."""
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(HF_MODEL_NAME)
    return pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=300,
    )


def run_model_comparison(queries: list[str]) -> None:
    """Run the same RAG prompt through both models for a list of queries."""
    groq_llm = load_groq_model()
    hf_pipe = load_hf_pipeline()

    for query in queries:
        docs = hybrid_retriever(query, top_k=5)
        context = build_context(docs)
        prompt = build_prompt(query, context, system_prompt=SYSTEM_PROMPT_1)

        groq_output = groq_llm.invoke(prompt).content
        hf_output = hf_pipe(prompt)[0]["generated_text"]

        print("=" * 80)
        print("QUERY:", query)
        print("\nPROMPT:\n", prompt)
        print("\nGROQ OUTPUT:\n", groq_output)
        print("\nHF OUTPUT:\n", hf_output)


def main() -> None:
    """Run the LLM comparison experiment for the fixed query set."""
    run_model_comparison(QUERIES)


if __name__ == "__main__":
    main()
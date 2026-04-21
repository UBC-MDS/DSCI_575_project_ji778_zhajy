import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

from src.hybrid import hybrid_retriever
from src.rag_pipeline import build_context
from src.prompts import build_prompt, SYSTEM_PROMPT_1

load_dotenv()

groq_llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)

model_id = "google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

hf_pipe = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=300,
)

queries = [
    "Glee season 4 musical comedy",
    "Taylor Cole Jack Turner chalet wedding",
    "Texas sheriff border corruption movie",
    "comedy about a guy going to wild parties in Las Vegas to find a girlfriend",
    "independent film about mental illness with an ending that lacks closure",
]

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
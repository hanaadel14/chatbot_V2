# services/generator.py

import os
from services.retriever import retrieve
from google.generativeai import configure, GenerativeModel
from deep_translator import GoogleTranslator

# Configure Gemini DREE API key
configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini = GenerativeModel(model_name="gemini-1.5-pro")


def generate_answer(query: str) -> str:
    """
    Given a user query (English or Arabic), retrieve relevant
    JSON records, call Gemini to generate a fluent answer,
    and translate back to Arabic if needed.
    """
    # 1) Retrieve relevant snippets & detect language
    data = retrieve(query)
    snippets = "\n\n".join(data["snippets"])
    lang = data["lang"]

    # 2) Build the LLM prompt
    prompt = (
        f"Use these records:\n{snippets}\n\n"
        f"Answer: {query}\n"
        "Keep any [PLATE]…[/PLATE] tags unchanged."
    )

    # 3) Generate with Gemini
    response = gemini.generate(
        prompt=prompt,
        temperature=0.2,
        max_output_tokens=512
    )
    answer = response.last.generations[0].text

    # 4) If original query was Arabic, translate answer back
    if lang == "ar":
        answer = GoogleTranslator(source="en", target="ar").translate(answer)

    return answer

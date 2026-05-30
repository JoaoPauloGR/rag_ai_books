import ollama

_PROMPT_TEMPLATE = """\
You are a helpful assistant answering questions about AI and machine learning.
Use only the context below to answer. If the answer is not in the context, say exactly:
"I don't have enough information in the retrieved passages to answer that."

Context:
{context}

Question: {question}

Answer:"""


def generate_answer(question: str, chunks: list[dict], generation_model: str) -> str:
    context = "\n\n---\n\n".join(chunk["text"] for chunk in chunks)
    prompt = _PROMPT_TEMPLATE.format(context=context, question=question)
    response = ollama.chat(
        model=generation_model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"].strip()

import ollama

_PROMPT_TEMPLATE = """\
Given the conversation so far and a follow-up question, rewrite the follow-up as a
single standalone question that can be understood without the conversation. Keep it
faithful to the user's intent. Do not answer it. Return only the rewritten question.

Conversation:
{history}

Follow-up: {question}

Standalone question:"""

_ROLE_LABELS = {"user": "User", "assistant": "Assistant"}


def rewrite_followup(history: list[dict], question: str, generation_model: str) -> str:
    if not history:
        return question

    rendered = "\n".join(
        f'{_ROLE_LABELS.get(turn["role"], turn["role"].capitalize())}: {turn["content"]}'
        for turn in history
    )
    prompt = _PROMPT_TEMPLATE.format(history=rendered, question=question)
    response = ollama.chat(
        model=generation_model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"].strip()

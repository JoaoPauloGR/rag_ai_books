# Group 3 — Generation (src/generate.py)
# ollama.chat is mocked; no running Ollama required.
from unittest.mock import patch


_CHUNKS = [
    {"text": "Attention allows models to focus on relevant tokens.", "source_file": "a.pdf", "page_number": 1},
    {"text": "Self-attention is computed across all positions.", "source_file": "b.pdf", "page_number": 4},
]


def _mock_chat(content="answer"):
    return {"message": {"content": content}}


@patch("src.generate.ollama.chat")
def test_returns_stripped_response(mock_chat):
    mock_chat.return_value = _mock_chat("  The answer.  ")
    from src.generate import generate_answer
    assert generate_answer("What is attention?", _CHUNKS, "llama3.2") == "The answer."


@patch("src.generate.ollama.chat")
def test_prompt_contains_all_chunk_texts(mock_chat):
    mock_chat.return_value = _mock_chat()
    from src.generate import generate_answer
    generate_answer("What is attention?", _CHUNKS, "llama3.2")

    prompt = mock_chat.call_args[1]["messages"][0]["content"]
    assert "Attention allows models to focus on relevant tokens." in prompt
    assert "Self-attention is computed across all positions." in prompt


@patch("src.generate.ollama.chat")
def test_prompt_contains_hard_constraint(mock_chat):
    mock_chat.return_value = _mock_chat()
    from src.generate import generate_answer
    generate_answer("What is attention?", _CHUNKS, "llama3.2")

    prompt = mock_chat.call_args[1]["messages"][0]["content"]
    assert "I don't have enough information" in prompt


@patch("src.generate.ollama.chat")
def test_uses_correct_generation_model(mock_chat):
    mock_chat.return_value = _mock_chat()
    from src.generate import generate_answer
    generate_answer("What is attention?", _CHUNKS, "llama3.1:8b")

    assert mock_chat.call_args[1]["model"] == "llama3.1:8b"

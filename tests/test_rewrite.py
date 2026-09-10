# Group 3 — Follow-up rewriting (src/rewrite.py)
# ollama.chat is mocked; no running Ollama required.
from unittest.mock import patch


_HISTORY = [
    {"role": "user", "content": "What is the transformer architecture?"},
    {"role": "assistant", "content": "It is a model built on self-attention."},
]


def _mock_chat(content="Standalone?"):
    return {"message": {"content": content}}


@patch("src.rewrite.ollama.chat")
def test_empty_history_returns_input_verbatim_and_makes_no_call(mock_chat):
    from src.rewrite import rewrite_followup
    result = rewrite_followup([], "What about its downsides?", "llama3.2")
    assert result == "What about its downsides?"
    mock_chat.assert_not_called()


@patch("src.rewrite.ollama.chat")
def test_non_empty_history_calls_ollama_once_and_returns_stripped_content(mock_chat):
    mock_chat.return_value = _mock_chat("  What are the downsides of the transformer?  ")
    from src.rewrite import rewrite_followup
    result = rewrite_followup(_HISTORY, "What about its downsides?", "llama3.2")
    assert result == "What are the downsides of the transformer?"
    mock_chat.assert_called_once()


@patch("src.rewrite.ollama.chat")
def test_prompt_contains_history_lines_and_followup(mock_chat):
    mock_chat.return_value = _mock_chat()
    from src.rewrite import rewrite_followup
    rewrite_followup(_HISTORY, "What about its downsides?", "llama3.2")

    prompt = mock_chat.call_args[1]["messages"][0]["content"]
    assert "User: What is the transformer architecture?" in prompt
    assert "Assistant: It is a model built on self-attention." in prompt
    assert "What about its downsides?" in prompt


@patch("src.rewrite.ollama.chat")
def test_uses_given_generation_model(mock_chat):
    mock_chat.return_value = _mock_chat()
    from src.rewrite import rewrite_followup
    rewrite_followup(_HISTORY, "And upsides?", "llama3.1:8b")
    assert mock_chat.call_args[1]["model"] == "llama3.1:8b"

import argparse
import sys
from functools import partial
from pathlib import Path
from uuid import uuid4

import chromadb.errors
import gradio as gr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.feedback import init_db, record_feedback, record_turn
from src.generate import generate_answer
from src.retrieve import retrieve_chunks
from src.rewrite import rewrite_followup

_NO_COLLECTION_MSG = (
    "I can't find the document collection yet. Run the ingestion pipeline "
    "(`uv run python src/ingest.py --dir <folder>`) and ask again."
)
_RATING_TO_DB = {"👍": "up", "👎": "down"}


def _citations(chunks: list[dict]) -> tuple[list[str], dict[str, list[int]]]:
    """Dedupe (source_file, page_number) preserving first-seen order.

    Returns the ``[Source: file, p. N]`` labels and a map from each label to the
    retrieval ranks it covers.
    """
    labels: list[str] = []
    ranks_by_label: dict[str, list[int]] = {}
    for rank, chunk in enumerate(chunks):
        label = f"[Source: {chunk['source_file']}, p. {chunk['page_number']}]"
        if label not in ranks_by_label:
            ranks_by_label[label] = []
            labels.append(label)
        ranks_by_label[label].append(rank)
    return labels, ranks_by_label


def on_message(cfg, user_msg, chat_history, session_id, last_turn):
    user_msg = (user_msg or "").strip()
    if not user_msg:
        return chat_history, "", gr.CheckboxGroup(), last_turn

    history = [{"role": m["role"], "content": m["content"]} for m in chat_history]
    if cfg["rewrite_followups"] and history:
        retrieval_q = rewrite_followup(history, user_msg, cfg["generation_model"])
    else:
        retrieval_q = user_msg

    try:
        chunks = retrieve_chunks(
            retrieval_q,
            cfg["embedding_model"],
            cfg["chroma_path"],
            cfg["collection_name"],
            cfg["top_k"],
        )
    except chromadb.errors.NotFoundError:
        chat_history = chat_history + [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": _NO_COLLECTION_MSG},
        ]
        return chat_history, "", gr.CheckboxGroup(choices=[], value=[]), last_turn

    answer = generate_answer(retrieval_q, chunks, cfg["generation_model"])
    citations, ranks_by_citation = _citations(chunks)

    turn_id = record_turn(
        cfg["feedback_db_path"],
        session_id=session_id,
        raw_question=user_msg,
        retrieval_question=retrieval_q,
        answer=answer,
        chunks=chunks,
        cfg=cfg,
    )

    rendered = answer
    if citations:
        rendered += "\n\n" + "\n".join(citations)
    chat_history = chat_history + [
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": rendered},
    ]

    new_last_turn = {
        "turn_id": turn_id,
        "citations": citations,
        "ranks_by_citation": ranks_by_citation,
    }
    return (
        chat_history,
        "",
        gr.CheckboxGroup(choices=citations, value=[]),
        new_last_turn,
    )


def on_feedback(cfg, rating, helpful_citations, comment, last_turn):
    if last_turn is None or not rating:
        gr.Warning("Ask a question and pick 👍 or 👎 before submitting feedback.")
        return comment, rating

    ranks: list[int] = []
    for citation in helpful_citations or []:
        ranks.extend(last_turn["ranks_by_citation"].get(citation, []))

    record_feedback(
        cfg["feedback_db_path"],
        turn_id=last_turn["turn_id"],
        rating=_RATING_TO_DB.get(rating, rating),
        comment=comment.strip() or None if comment else None,
        helpful_ranks=sorted(set(ranks)),
    )
    gr.Info("Feedback saved")
    return "", None


def build_demo(cfg) -> gr.Blocks:
    with gr.Blocks(title="RAG chat") as demo:
        session_id = gr.State()
        last_turn = gr.State(None)

        # Gradio 6 uses the openai-style message format for Chatbot unconditionally
        # (the pre-6 `type="messages"` kwarg was removed).
        chatbot = gr.Chatbot(label="Conversation", height=460)
        with gr.Row():
            msg = gr.Textbox(
                placeholder="Ask a question about the ingested documents…",
                show_label=False,
                scale=8,
            )
            send = gr.Button("Send", variant="primary", scale=1)

        with gr.Accordion("Feedback on the last answer", open=True):
            rating = gr.Radio(["👍", "👎"], label="Rating")
            helpful = gr.CheckboxGroup([], label="Which sources helped?")
            comment = gr.Textbox(label="Comment", lines=2)
            submit_feedback = gr.Button("Submit feedback")

        handle_message = partial(on_message, cfg)
        handle_feedback = partial(on_feedback, cfg)
        message_io = (
            handle_message,
            [msg, chatbot, session_id, last_turn],
            [chatbot, msg, helpful, last_turn],
        )
        msg.submit(*message_io)
        send.click(*message_io)
        submit_feedback.click(
            handle_feedback,
            [rating, helpful, comment, last_turn],
            [comment, rating],
        )

        demo.load(lambda: str(uuid4()), None, session_id)
    return demo


def main():
    parser = argparse.ArgumentParser(description="Launch the RAG chat UI.")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--port", type=int, default=7860, help="Port to serve on")
    args = parser.parse_args()

    cfg = load_config(args.config)
    init_db(cfg["feedback_db_path"])
    build_demo(cfg).launch(
        server_name="127.0.0.1", server_port=args.port, share=False
    )


if __name__ == "__main__":
    main()

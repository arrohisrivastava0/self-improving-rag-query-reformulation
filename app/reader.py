import re

from app.config import GEN_MODEL
from app.llm import chat

SYSTEM = "You answer multi-hop factoid questions using ONLY the provided passages."

PROMPT = """Passages:
{passages}

Question: {question}

Rules:
- Use only the passages above. If they seem incomplete, give your best guess from them.
- Reply with the answer ONLY: a short phrase, name, date or number (usually 1-5 words).
- For yes/no questions reply exactly "yes" or "no".
- No explanation and no full sentences.

Answer:"""


SYSTEM_CB = "You answer multi-hop factoid questions from your own knowledge."

PROMPT_CB = """Question: {question}

Rules:
- Answer from your own knowledge. If unsure, give your best guess.
- Reply with the answer ONLY: a short phrase, name, date or number (usually 1-5 words).
- For yes/no questions reply exactly "yes" or "no".
- No explanation and no full sentences.

Answer:"""


def _clean(text):
    text = text.strip()
    line = text.splitlines()[0] if text else ""
    line = re.sub(r"^\s*(final\s+)?answer\s*:\s*", "", line, flags=re.I)
    return line.strip().strip('"').strip("*").rstrip(".").strip()


def answer_question(question, passages, model=GEN_MODEL):
    """passages: list of dicts with a 'text' field. Returns (short_answer, LLMResponse).
    With an empty list it runs closed-book (no retrieval) - a standard control experiment."""
    if not passages:
        resp = chat(PROMPT_CB.format(question=question), system=SYSTEM_CB, model=model, max_tokens=768)
        return _clean(resp.text), resp
    block = "\n".join(f"[{i + 1}] {p['text']}" for i, p in enumerate(passages))
    resp = chat(
        PROMPT.format(passages=block, question=question),
        system=SYSTEM,
        model=model,
        max_tokens=768,
    )
    return _clean(resp.text), resp

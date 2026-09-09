"""Review-1 smoke test: proves the local + cloud stack is live.

Checks (1) local embeddings via Ollama and (2) both Groq models used in the design.
Run from the project root with the venv active:   python test_setup.py
"""
import os
import sys
import time

from dotenv import load_dotenv

load_dotenv()

from app.config import AGENT_MODEL, EMBED_MODEL, GEN_MODEL  # noqa: E402


def check_embeddings():
    from app.embed import embed_query

    v = embed_query("ping")
    print(f"OK    local embeddings ({EMBED_MODEL} via Ollama), dim = {len(v)}")


def check_groq(model):
    from groq import Groq

    key = os.environ.get("GROQ_API_KEY")
    if not key or key == "your_key_here":
        raise RuntimeError("GROQ_API_KEY missing - copy .env.example to .env and paste your key")
    t = time.time()
    res = Groq(api_key=key).chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Reply with the single word: pong"}],
        max_completion_tokens=512,  # gpt-oss models 'think' first; leave room for the answer
    )
    reply = (res.choices[0].message.content or "").strip()
    print(f"OK    Groq {model}: {reply!r}  ({time.time() - t:.1f}s)")


def main():
    print("Review 1 environment validation\n")
    failures = 0
    for name, fn in [
        ("embeddings", check_embeddings),
        (AGENT_MODEL, lambda: check_groq(AGENT_MODEL)),
        (GEN_MODEL, lambda: check_groq(GEN_MODEL)),
    ]:
        try:
            fn()
        except Exception as e:
            failures += 1
            print(f"FAIL  {name}: {e}")
    print("\nAll checks passed." if not failures else f"\n{failures} check(s) failed.")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()

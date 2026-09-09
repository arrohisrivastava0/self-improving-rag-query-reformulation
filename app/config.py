from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
QUESTIONS_PATH = DATA_DIR / "questions.json"
CORPUS_PATH = DATA_DIR / "corpus.jsonl"
FAISS_PATH = DATA_DIR / "faiss.index"
FAISS_IDS_PATH = DATA_DIR / "faiss_ids.json"      # doc_id for every row of the index
RETRIEVAL_CHECK_PATH = DATA_DIR / "retrieval_check.json"
LLM_CACHE_PATH = DATA_DIR / "llm_cache.sqlite"    # every Groq response is cached here
RUNS_DIR = DATA_DIR / "runs"                      # one .jsonl per experiment run

N_QUESTIONS = 200          # evaluation questions (same 200 as Day 1 for the same SEED)
N_CORPUS_QUESTIONS = 1000  # corpus is pooled from this many questions -> ~10k paragraphs
SEED = 42

# Local embeddings (Ollama) - small, runs on CPU
EMBED_MODEL = "nomic-embed-text"

# Groq limits are counted PER MODEL, so spread the load across two models:
AGENT_MODEL = "openai/gpt-oss-20b"    # router, checklist gating, candidate rewrites
GEN_MODEL = "openai/gpt-oss-120b"     # final answer synthesis (and baseline reader)

REASONING_EFFORT = "low"   # gpt-oss 'thinking' budget: low = cheaper + faster
# Requests/minute per model. Check console.groq.com -> Limits and put YOUR numbers here.
GROQ_RPM = {AGENT_MODEL: 30, GEN_MODEL: 30}
MAX_RETRIES = 6

TOP_K = 5

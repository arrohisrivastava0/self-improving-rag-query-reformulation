from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
QUESTIONS_PATH = DATA_DIR / "questions.json"
CORPUS_PATH = DATA_DIR / "corpus.jsonl"
FAISS_PATH = DATA_DIR / "faiss.index"
FAISS_IDS_PATH = DATA_DIR / "faiss_ids.json"     
RETRIEVAL_CHECK_PATH = DATA_DIR / "retrieval_check.json"
LLM_CACHE_PATH = DATA_DIR / "llm_cache.sqlite"    
RUNS_DIR = DATA_DIR / "runs"                      

N_QUESTIONS = 200         
N_CORPUS_QUESTIONS = 1000
SEED = 42

EMBED_MODEL = "nomic-embed-text"

AGENT_MODEL = "openai/gpt-oss-20b"    
GEN_MODEL = "openai/gpt-oss-120b"     

REASONING_EFFORT = "low"   
GROQ_RPM = {AGENT_MODEL: 30, GEN_MODEL: 30}
MAX_RETRIES = 6

TOP_K = 5

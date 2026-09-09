# Self-Improving RAG via Automatic Query Reformulation - Review 1 codebase

Stack: Python venv - LangGraph (later) - FAISS (local index) - Ollama nomic-embed-text (local embeddings)
       - Groq gpt-oss-20b / gpt-oss-120b (cloud LLM calls). No Docker, no GPU.

## One-time setup (Windows PowerShell)
1. Install Ollama (https://ollama.com/download), then:
   ```
   ollama pull nomic-embed-text
   ```
   (Do NOT pull an 8B chat model - LLM calls go to Groq.)
2. Get a free Groq API key at https://console.groq.com
3. From the project root:
   ```
   py -3.11 -m venv .venv
   .venv\Scripts\Activate.ps1        # if blocked: Set-ExecutionPolicy -Scope Process Bypass
   pip install -r requirements.txt
   copy .env.example .env             # then open .env and paste your key
   ```

## Day 1 pipeline (project root, in this order)
```
python test_setup.py                    # embeddings + both Groq models respond
python -m app.index.prepare_data        # samples 200 HotpotQA questions -> data/corpus.jsonl
python -m app.index.build_index         # embeds + builds FAISS index (~2-5 min)
python -m app.retrieval "your query"    # prints top-5 passages
python -m app.eval.retrieval_check      # recall of gold paragraphs with the raw question
```

## Layout
- `test_setup.py`         Review-1 smoke test (run it live in the demo)
- `app/config.py`         all knobs (subset size, model names, paths)
- `app/embed.py`          the only place embeddings are created (query/doc prefixes)
- `app/retrieval.py`      `retrieve(query, k)` used by baseline + pipeline
- `app/index/`            data prep + FAISS index build
- `app/eval/`             evaluation scripts (EM/F1 harness comes Day 2)

## Groq quota tips
Limits are per model. Agent nodes use gpt-oss-20b, generation uses gpt-oss-120b.
Check your live limits at console.groq.com -> Limits. Cache every LLM response (Day 2).

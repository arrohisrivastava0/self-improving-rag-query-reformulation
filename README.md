# Self-Improving RAG through Automatic Query Reformulation

Implementation of a Retrieval-Augmented Generation (RAG) system that iteratively improves retrieval by checking whether the retrieved evidence is sufficient and reformulating the query when necessary.

## Project Overview

The project compares a standard RAG baseline with an adaptive retrieval pipeline.

### Baseline RAG

```text
Question
   ↓
Embedding
   ↓
FAISS Retrieval
   ↓
Top-k Evidence
   ↓
LLM
   ↓
Answer
```

### Adaptive RAG

```text
Question
   ↓
Initial Retrieval
   ↓
Sufficiency Check
   ↓
 ┌───────────────┐
 │ Evidence      │
 │ sufficient?   │
 └───────┬───────┘
         │
    Yes  │  No
         │
         │
         ↓
   Query Reformulation
         ↓
   Retrieve Again
         ↓
   Sufficiency Check
         ↓
      ...
         ↓
   Final Evidence
         ↓
        LLM
         ↓
       Answer
```

The system performs inference-time iterative retrieval. It does not update model weights during inference.

## Technology Stack

- Python 3.11
- FAISS for local vector retrieval
- Ollama with `nomic-embed-text` for local embeddings
- Groq API for LLM inference
  - `openai/gpt-oss-20b` for agent/checklist and query-reformulation tasks
  - `openai/gpt-oss-120b` for final answer generation
- HotpotQA for the question and evidence corpus
- Python virtual environment

No GPU is required for the embedding/retrieval pipeline.

## Project Structure

```text
.
├── app/
│   ├── config.py
│   ├── embed.py
│   ├── llm.py
│   ├── pipeline.py
│   ├── prompts.py
│   ├── reader.py
│   ├── retrieval.py
│   │
│   ├── index/
│   │   ├── build_index.py
│   │   └── prepare_data.py
│   │
│   └── eval/
│       ├── metrics.py
│       ├── score.py
│       ├── trace_format.py
│       ├── run_baseline.py
│       ├── run_pipeline.py
│       ├── analyze_traces.py
│       ├── diagnose.py
│       ├── diagnose_query_limited.py
│       ├── compare_recovery.py
│       ├── overlap.py
│       ├── check_router.py
│       ├── list_failures.py
│       ├── peek.py
│       └── show_trace.py
│
├── requirements.txt
├── test_setup.py
└── README.md
```

Generated data, model caches, API credentials, virtual environments, and local backups are excluded through `.gitignore`.

## Setup

### 1. Create the Python environment

From the project root:

```bash
py -3.11 -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### 2. Install Ollama and the embedding model

Install Ollama and pull the local embedding model:

```bash
ollama pull nomic-embed-text
```

The project uses Ollama for embeddings. LLM generation is performed through Groq.

### 3. Configure the Groq API key

Create a `.env` file in the project root:

```text
GROQ_API_KEY=your_api_key_here
```

Do not commit `.env` to Git.

## Building the Corpus and Index

Prepare the HotpotQA subset:

```bash
python -m app.index.prepare_data
```

Build the FAISS index:

```bash
python -m app.index.build_index
```

The generated corpus and FAISS files are stored locally under `data/` and are intentionally excluded from Git.

## Evaluation

The repository contains separate evaluation paths for the baseline and adaptive pipeline.

Baseline evaluation:

```bash
python -m app.eval.run_baseline
```

Adaptive pipeline evaluation:

```bash
python -m app.eval.run_pipeline
```

Additional scripts are provided for trace inspection, recovery analysis, failure classification, retrieval overlap analysis, and query-limited diagnostics.

## Final Held-Out Test Results

The held-out test set contains 100 questions.

| System | EM | F1 | Relaxed Accuracy | All-Gold Retrieval | Tokens / Question |
|---|---|---|---|---|---|
| Baseline k=5 | 0.530 | 0.681 | 0.720 | 0.660 | 866 |
| Baseline k=10 | 0.550 | 0.703 | 0.750 | 0.770 | 1587 |
| Adaptive pipeline | 0.590 | 0.720 | 0.780 | 0.770 | 2816 |

The adaptive pipeline recovered all required gold paragraphs in 11 of the 34 questions where the initial k=5 retrieval missed at least one gold paragraph.

No initial k=5 retrieval success was lost by the adaptive pipeline in the held-out test.

The k=10 baseline and adaptive pipeline retrieved all gold paragraphs at the same observed rate (0.770), while their answer-level results differed on this test set.

## Failure Analysis

Among the 34 initial k=5 retrieval failures:

- 11 (32%) were recovered by the adaptive pipeline.
- 1 (3%) was limited by the final evidence cut.
- 7 (21%) were cases where the sufficiency gate stopped despite missing gold evidence.
- 3 (9%) were selector-limited cases where a better candidate query existed.
- 12 (35%) were query-limited cases where none of the attempted candidate queries retrieved the missing gold paragraph in the required evidence.

A further offline analysis of the 12 query-limited cases found:

- 4 had the missing gold paragraph within the top 10 for a tested query.
- 4 placed the missing paragraph between ranks 11 and 50.
- 4 did not retrieve the missing paragraph within the top 50 for any attempted query.

These diagnostics are used to identify directions for further improvement rather than to claim that any single component is solely responsible for the observed failures.

## Metrics

The evaluation reports:

- Exact Match (EM)
- Token-level F1
- Relaxed answer accuracy
- All-gold retrieval rate
- Retrieval/generation token usage

The relaxed accuracy metric is a project-specific evaluation metric and should not be interpreted as the official HotpotQA evaluation metric.

## Limitations

The current evaluation is based on:

- HotpotQA
- A 100-question held-out test set
- One embedding model
- One primary LLM configuration
- A local FAISS retrieval index
- A single adaptive pipeline configuration

The adaptive pipeline also uses more tokens per question than the fixed-depth baselines. Further work therefore includes improving the sufficiency gate, making retrieval depth adaptive, improving query reformulation for semantic retrieval gaps, and investigating learned candidate selection.

## License

This repository is intended for academic/thesis research.

---

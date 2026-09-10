"""Sample a HotpotQA (distractor setting) subset and build a pooled corpus.

Outputs:
  data/questions.json  - the EVALUATION questions (gold answers + gold supporting titles)
  data/corpus.jsonl    - de-duplicated paragraphs pooled from the evaluation questions PLUS
                         extra questions, so the corpus is ~10k paragraphs and retrieval is
                         realistically hard (other questions' paragraphs act as distractors).

The evaluation questions are drawn exactly as on Day 1, so they are the SAME 200 questions.

Run:  python -m app.index.prepare_data [--n 200] [--corpus-n 1000]
"""
import argparse
import hashlib
import json
import random
import urllib.request
from collections import Counter

from app.config import CORPUS_PATH, DATA_DIR, N_CORPUS_QUESTIONS, N_QUESTIONS, QUESTIONS_PATH, SEED

CMU_URL = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"


def _load_from_hf():
    from datasets import load_dataset

    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    rows = []
    for r in ds:
        rows.append(
            {
                "id": r["id"],
                "question": r["question"],
                "answer": r["answer"],
                "type": r["type"],
                "level": r["level"],
                "supporting_titles": sorted(set(r["supporting_facts"]["title"])),
                "context": list(zip(r["context"]["title"], r["context"]["sentences"])),
            }
        )
    return rows


def _load_from_cmu():
    print(f"Downloading {CMU_URL} (~46 MB) ...")
    with urllib.request.urlopen(CMU_URL) as f:
        raw = json.load(f)
    rows = []
    for r in raw:
        rows.append(
            {
                "id": r["_id"],
                "question": r["question"],
                "answer": r["answer"],
                "type": r["type"],
                "level": r["level"],
                "supporting_titles": sorted({t for t, _ in r["supporting_facts"]}),
                "context": [(t, s) for t, s in r["context"]],
            }
        )
    return rows


def load_raw():
    try:
        return _load_from_hf()
    except Exception as e:  # network / datasets-version problems
        print(f"HF load failed ({e!r}); falling back to direct download.")
        return _load_from_cmu()


def build(n_eval=N_QUESTIONS, n_corpus=N_CORPUS_QUESTIONS, seed=SEED):
    rows = load_raw()
    print(f"Loaded {len(rows)} validation questions.")
    n_corpus = max(n_corpus, n_eval)

    # Same call as Day 1 -> identical evaluation questions.
    sample = random.Random(seed).sample(rows, n_eval)
    eval_ids = {r["id"] for r in sample}
    rest = [r for r in rows if r["id"] not in eval_ids]
    extra = random.Random(seed + 1).sample(rest, n_corpus - n_eval)

    corpus = {}
    for r in sample + extra:
        for title, sentences in r["context"]:
            text = f"{title}. " + " ".join(s.strip() for s in sentences)
            doc_id = hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
            corpus.setdefault(doc_id, {"doc_id": doc_id, "title": title, "text": text})

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    questions = [
        {k: r[k] for k in ("id", "question", "answer", "type", "level", "supporting_titles")}
        for r in sample
    ]
    QUESTIONS_PATH.write_text(json.dumps(questions, indent=2, ensure_ascii=False), encoding="utf-8")
    with CORPUS_PATH.open("w", encoding="utf-8") as f:
        for d in corpus.values():
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    print(f"Evaluation questions: {len(questions)}  ->  {QUESTIONS_PATH}")
    print(f"  by type : {dict(Counter(q['type'] for q in questions))}")
    print(f"Corpus built from {n_corpus} questions: {len(corpus)} unique paragraphs  ->  {CORPUS_PATH}")
    print("NEXT: python -m app.index.build_index   (re-embeds everything, ~10-20 min)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=N_QUESTIONS, help="evaluation questions")
    ap.add_argument("--corpus-n", type=int, default=N_CORPUS_QUESTIONS, help="questions pooled into the corpus")
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args()
    build(a.n, a.corpus_n, a.seed)

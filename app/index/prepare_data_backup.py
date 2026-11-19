"""Sample a HotpotQA (distractor setting) subset and build a pooled corpus.

Outputs:
  data/questions.json  - sampled questions with gold answers + gold supporting titles
  data/corpus.jsonl    - de-duplicated paragraphs pooled from ALL sampled questions

Run:  python -m app.index.prepare_data [--n 200]
"""
import argparse
import hashlib
import json
import random
import urllib.request
from collections import Counter

from app.config import CORPUS_PATH, DATA_DIR, N_QUESTIONS, QUESTIONS_PATH, SEED

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


def build(n=N_QUESTIONS, seed=SEED):
    rows = load_raw()
    print(f"Loaded {len(rows)} validation questions.")
    sample = random.Random(seed).sample(rows, n)

    corpus = {}
    for r in sample:
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

    print(f"Questions: {len(questions)}  ->  {QUESTIONS_PATH}")
    print(f"  by type : {dict(Counter(q['type'] for q in questions))}")
    print(f"  by level: {dict(Counter(q['level'] for q in questions))}")
    print(f"Corpus paragraphs (deduped): {len(corpus)}  ->  {CORPUS_PATH}")
    print("NOTE: if you re-run this, re-run app.index.build_index too.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=N_QUESTIONS)
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args()
    build(a.n, a.seed)

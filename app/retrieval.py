"""Shared retrieval function. Every later node (baseline, retrieve_node) imports this.

Try it:  python -m app.retrieval "Which magazine was started first, Arthur's Magazine or First for Women?"
"""
import json
import sys
from functools import lru_cache

import faiss
import numpy as np

from app.config import CORPUS_PATH, FAISS_IDS_PATH, FAISS_PATH, TOP_K
from app.embed import embed_query


@lru_cache(maxsize=1)
def _load():
    index = faiss.deserialize_index(np.frombuffer(FAISS_PATH.read_bytes(), dtype="uint8"))
    ids = json.loads(FAISS_IDS_PATH.read_text(encoding="utf-8"))
    corpus = {}
    for line in CORPUS_PATH.open(encoding="utf-8"):
        d = json.loads(line)
        corpus[d["doc_id"]] = d
    try:
        docs = [corpus[i] for i in ids]
    except KeyError:
        raise RuntimeError("Index and corpus are out of sync - re-run: python -m app.index.build_index")
    if index.ntotal != len(docs):
        raise RuntimeError("Index and corpus are out of sync - re-run: python -m app.index.build_index")
    return index, docs


def retrieve(query: str, k: int = TOP_K):
    """Return top-k passages as dicts: doc_id, title, text, score (cosine similarity; higher = closer)."""
    index, docs = _load()
    q = np.asarray([embed_query(query)], dtype="float32")
    faiss.normalize_L2(q)
    scores, idxs = index.search(q, k)
    return [
        {
            "doc_id": docs[i]["doc_id"],
            "title": docs[i]["title"],
            "text": docs[i]["text"],
            "score": float(s),
        }
        for s, i in zip(scores[0], idxs[0])
        if i != -1
    ]


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "Which magazine was started first, Arthur's Magazine or First for Women?"
    for h in retrieve(q):
        print(f"[{h['score']:.3f}] {h['title']}\n    {h['text'][:160]}...\n")

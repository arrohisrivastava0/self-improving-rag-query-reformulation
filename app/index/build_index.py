"""Embed the pooled corpus and store it as a FAISS index.

Run:  python -m app.index.build_index
Safe to re-run: it rebuilds the index from scratch.

Design notes
- IndexFlatIP on L2-normalised vectors = exact cosine search (fine for a few thousand docs).
- The index is saved as raw bytes via serialize_index, because faiss.write_index can fail
  on Windows paths containing non-ASCII characters (e.g. a user folder with accents).
"""
import json

import faiss
import numpy as np
from tqdm import tqdm

from app.config import CORPUS_PATH, FAISS_IDS_PATH, FAISS_PATH
from app.embed import embed_documents

BATCH = 32


def main():
    docs = [json.loads(line) for line in CORPUS_PATH.open(encoding="utf-8")]
    print(f"Embedding {len(docs)} paragraphs with Ollama ...")

    chunks = []
    for i in tqdm(range(0, len(docs), BATCH)):
        batch = docs[i : i + BATCH]
        chunks.append(np.asarray(embed_documents([d["text"] for d in batch]), dtype="float32"))
    X = np.vstack(chunks)
    faiss.normalize_L2(X)

    index = faiss.IndexFlatIP(X.shape[1])
    index.add(X)

    FAISS_PATH.write_bytes(faiss.serialize_index(index).tobytes())
    FAISS_IDS_PATH.write_text(json.dumps([d["doc_id"] for d in docs]), encoding="utf-8")
    print(f"Done. {index.ntotal} vectors (dim {X.shape[1]}) -> {FAISS_PATH}")


if __name__ == "__main__":
    main()

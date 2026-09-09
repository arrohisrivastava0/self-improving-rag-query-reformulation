"""Embedding helpers (Ollama + nomic-embed-text).

nomic-embed-text is trained with task prefixes, so documents and queries
must use different prefixes. Keep these two functions as the ONLY place
embeddings are created, so index and query time always match.
"""
import ollama

from app.config import EMBED_MODEL


def _embed(inputs):
    resp = ollama.embed(model=EMBED_MODEL, input=inputs)
    return resp["embeddings"]


def embed_documents(texts):
    return _embed(["search_document: " + t for t in texts])


def embed_query(text):
    return _embed(["search_query: " + text])[0]

# services/retriever.py

import os
import jsonlines
import numpy as np
from sentence_transformers import SentenceTransformer
from langdetect import detect
from deep_translator import GoogleTranslator
from pathlib import Path
import torch

# Paths
ROOT = Path(__file__).parent.parent
corpus_path = ROOT / "data_preprocessed" / "corpus.jsonl"
emb_path    = ROOT / "data_preprocessed" / "embeddings.npy"

# Load corpus
print(f"[Retriever] Loading corpus from {corpus_path}")
docs = list(jsonlines.open(corpus_path))
texts = [d["text"] for d in docs]

# Load embeddings
print(f"[Retriever] Loading embeddings from {emb_path}")
embs = np.load(emb_path)
# Normalize
embs_norm = embs / np.linalg.norm(embs, axis=1, keepdims=True)

# Load embedding model
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[Retriever] Embedding model on {device}")
model = SentenceTransformer("all-MiniLM-L6-v2", device=device)


def retrieve(query: str, k: int = 5) -> dict:
    """
    Perform in-memory k-NN retrieval on the precomputed embeddings.
    Returns top-k text snippets and detected language.
    """
    # 1) Language detection + translate if Arabic
    lang = detect(query)
    q_en = GoogleTranslator(source="auto", target="en").translate(query) if lang == "ar" else query

    # 2) Embed query
    q_emb = model.encode([q_en], convert_to_numpy=True)[0]
    q_norm = q_emb / np.linalg.norm(q_emb)

    # 3) Cosine similarity
    sims = embs_norm.dot(q_norm)
    topk_idx = np.argsort(-sims)[:k]

    # 4) Return snippets and language
    return {
        "snippets": [texts[i] for i in topk_idx],
        "lang": lang
    }

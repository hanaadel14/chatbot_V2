# scripts/build_local_index.py

import os
import jsonlines
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
import json
import torch

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Paths
ROOT        = Path(__file__).parent.parent
corpus_path = ROOT / "data_preprocessed" / "corpus.jsonl"
emb_path    = ROOT / "data_preprocessed" / "embeddings.npy"
ids_path    = ROOT / "data_preprocessed" / "ids.json"

# 1) Load the flattened corpus
print(f"Loading corpus from {corpus_path}...")
docs  = list(jsonlines.open(corpus_path))
texts = [d["text"] for d in docs]
ids   = [d["id"]   for d in docs]
print(f"Found {len(ids)} documents.")

# 2) Choose device and load model
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

# 3) Compute embeddings
print("Computing embeddings (this may take a moment)...")
embs = model.encode(
    texts,
    batch_size=len(texts),        # one batch for small corpora
    show_progress_bar=True,
    convert_to_numpy=True
)

# 4) Save embeddings & IDs
print(f"Saving embeddings to {emb_path} and IDs to {ids_path}...")
np.save(emb_path, embs)
with open(ids_path, "w", encoding="utf-8") as f:
    json.dump(ids, f, ensure_ascii=False, indent=2)
print("✅ Saved embeddings and IDs.")

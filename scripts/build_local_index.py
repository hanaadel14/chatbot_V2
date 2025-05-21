# scripts/build_local_index.py
import os
# Suppress TensorFlow logs if present
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import jsonlines
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
import json
import torch

# Paths
ROOT        = Path(__file__).parent.parent
corpus_path = ROOT / "data_preprocessed" / "corpus.jsonl"
emb_path    = ROOT / "data_preprocessed" / "embeddings.npy"
ids_path    = ROOT / "data_preprocessed" / "ids.json"

# 1) Load the flattened corpus
print(f"Loading corpus from {corpus_path}...")
docs = list(jsonlines.open(corpus_path))
texts = [d["text"] for d in docs]
ids   = [d["id"]   for d in docs]
print(f"Found {len(ids)} documents.")

# 2) Set device for embedding
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

# 3) Compute embeddings
print("Computing embeddings (this may take a moment)...")
embs = model.encode(
    texts,
    batch_size=8,
    show_progress_bar=True,
    convert_to_numpy=True
)

# 4) Save embeddings & IDs
print(f"Saving embeddings to {emb_path} and IDs to {ids_path}...")
np.save(emb_path, embs)
with open(ids_path, "w", encoding="utf-8") as f:
    json.dump(ids, f, ensure_ascii=False, indent=2)
print(f"Saved {len(ids)} embeddings.")


# services/retriever/app.py
import os, json
import numpy as np
from sentence_transformers import SentenceTransformer
from langdetect import detect
from deep_translator import GoogleTranslator
from pathlib import Path
import jsonlines

# Load resources once
ROOT     = Path(__file__).parent.parent
corpus   = list(jsonlines.open(ROOT/"data_preprocessed"/"corpus.jsonl"))
texts    = [d["text"] for d in corpus]
ids      = [d["id"]   for d in corpus]
embs     = np.load(ROOT/"data_preprocessed"/"embeddings.npy")
# normalize for cosine similarity
embs_norm= embs / np.linalg.norm(embs, axis=1, keepdims=True)
model    = SentenceTransformer("all-MiniLM-L6-v2", device=device)


def retrieve(query, k=5):
    # 1) Detect language & translate if Arabic
    lang = detect(query)
    q_en = GoogleTranslator(source="auto", target="en").translate(query) if lang == "ar" else query

    # 2) Embed query
    q_emb  = model.encode([q_en], convert_to_numpy=True)[0]
    q_norm = q_emb / np.linalg.norm(q_emb)

    # 3) Cosine similarities & top-k
    sims = embs_norm.dot(q_norm)
    idxs = np.argsort(-sims)[:k]

    # 4) Return snippets & language
    return {"snippets": [texts[i] for i in idxs], "lang": lang}


def lambda_handler(event, context=None):
    return retrieve(event.get('query', ''))


# services/generator/app.py
import os
from google.generativeai import configure, GenerativeModel
from deep_translator import GoogleTranslator
# Import retriever
from services.retriever import retrieve

# Configure Gemini DREE
configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini = GenerativeModel(model_name="gemini-1.5-pro")


def generate_answer(user_q):
    # 1) Retrieve context
    data  = retrieve(user_q)
    snips = "\n\n".join(data['snippets'])

    # 2) Build prompt
    prompt = (
        f"Use these records:\n{snips}\n\n"
        f"Answer: {user_q}\n"
        "Keep any [PLATE]…[/PLATE] tags unchanged."
    )

    # 3) Generate via Gemini
    resp = gemini.generate(
        prompt=prompt,
        temperature=0.2,
        max_output_tokens=512
    )
    ans = resp.last.generations[0].text

    # 4) Translate back if needed
    if data['lang'] == 'ar':
        ans = GoogleTranslator(source='en', target='ar').translate(ans)

    return ans

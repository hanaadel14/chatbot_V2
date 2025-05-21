import json
from pathlib import Path
import jsonlines

# Determine project root and raw data directory
ROOT = Path(__file__).parent.parent
RAW_DIR = ROOT / "data_raw"
OUT_DIR = ROOT / "data_preprocessed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Debug: list raw directory contents
print(f"Looking for JSON files in: {RAW_DIR}")
entries = list(RAW_DIR.iterdir())
print(f"Entries in data_raw: {[e.name for e in entries]}")

# Collect and flatten JSON records (recursive search)
docs = []
for json_file in RAW_DIR.rglob("*.json"):
    try:
        rec = json.load(open(json_file, encoding="utf-8"))
    except Exception as e:
        print(f"Skipping {json_file}, error: {e}")
        continue

    # Preserve the vehicle plate with tags
    plate = rec.get("NAME", "")
    rec["NAME"] = f"[PLATE]{plate}[/PLATE]"

    # Flatten into a single text blob
    text = "  ".join(f"{k}: {v}" for k, v in rec.items())
    docs.append({"id": json_file.stem, "text": text})

# Write out to JSONL for indexing
output_file = OUT_DIR / "corpus.jsonl"
with jsonlines.open(output_file, "w") as writer:
    writer.write_all(docs)

print(f"Wrote {len(docs)} documents to {output_file}")

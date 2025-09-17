# app/chunk_and_embed.py
import os
from sentence_transformers import SentenceTransformer
import numpy as np
import json
from pathlib import Path
from app.utils import chunk_text

BASE = Path(__file__).resolve().parents[1] / "app" / "data"
MODEL_NAME = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
OUT_JSONL = BASE / "docs_parsed.jsonl"
EMBED_NPY = BASE / "embeddings.npy"

def load_text(txt_path):
    return Path(txt_path).read_text(encoding="utf-8")

def main(txt_path=None, chunk_size=500, chunk_overlap=50):
    if txt_path is None:
        txt_path = BASE / "HR_Policy_raw.txt"
    text = load_text(txt_path)
    chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    model = SentenceTransformer(MODEL_NAME)
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    # save jsonl with metadata
    with open(OUT_JSONL, "w", encoding="utf-8") as fw:
        for c, emb in zip(chunks, embeddings):
            rec = {
                "id": c["id"],
                "text": c["text"],
                "start": c["start"],
                "end": c["end"]
            }
            fw.write(json.dumps(rec, ensure_ascii=False) + "\n")
    np.save(EMBED_NPY, embeddings)
    print(f"Saved {len(chunks)} chunks to {OUT_JSONL} and embeddings to {EMBED_NPY}")

if __name__ == "__main__":
    main()

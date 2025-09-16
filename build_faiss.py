# app/build_faiss.py
import faiss
import numpy as np
from pathlib import Path
import json

BASE = Path(__file__).resolve().parents[1] / "app" / "data"
EMBED_NPY = BASE / "embeddings.npy"
JSONL = BASE / "docs_parsed.jsonl"
FAISS_PATH = BASE / "faiss_index.faiss"
META_PATH = BASE / "metadata.json"

def load_metadata(jsonl):
    metas = []
    with open(jsonl, "r", encoding="utf-8") as fr:
        for line in fr:
            metas.append(json.loads(line))
    return metas

def main():
    embs = np.load(EMBED_NPY)
    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product (we'll normalize embeddings)
    # normalize embeddings for cosine similarity
    faiss.normalize_L2(embs)
    index.add(embs)
    faiss.write_index(index, str(FAISS_PATH))
    metas = load_metadata(JSONL)
    with open(META_PATH, "w", encoding="utf-8") as fw:
        json.dump(metas, fw, ensure_ascii=False, indent=2)
    print(f"Saved FAISS index to {FAISS_PATH} and metadata to {META_PATH}")

if __name__ == "__main__":
    main()

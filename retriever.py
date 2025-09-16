# app/retriever.py
import os
import time
import json
from pathlib import Path
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import shelve
from typing import List, Dict

BASE = Path(__file__).resolve().parents[1] / "app" / "data"
FAISS_PATH = BASE / "faiss_index.faiss"
META_PATH = BASE / "metadata.json"
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CACHE_FILE = BASE / "query_cache.db"
TOP_K = int(os.getenv("TOP_K", 5))
CACHE_TTL = int(os.getenv("CACHE_TTL", 3600))  # seconds

class Retriever:
    def __init__(self):
        self.index = faiss.read_index(str(FAISS_PATH))
        with open(META_PATH, "r", encoding="utf-8") as fr:
            self.metas = json.load(fr)
        self.texts = [m["text"] for m in self.metas]
        self.bm25 = BM25Okapi([t.split() for t in self.texts])
        self.model = SentenceTransformer(EMBED_MODEL)
        self.cache = shelve.open(str(CACHE_FILE), writeback=True)

    def _cache_get(self, key):
        if key in self.cache:
            rec = self.cache[key]
            if time.time() - rec["ts"] < CACHE_TTL:
                return rec["value"]
            else:
                del self.cache[key]
        return None

    def _cache_set(self, key, value):
        self.cache[key] = {"ts": time.time(), "value": value}
        self.cache.sync()

    def retrieve(self, query: str, top_k: int = TOP_K) -> List[Dict]:
        """
        1) Try cache
        2) Embed query + search FAISS
        3) Get candidate texts, run BM25 reranking on candidates
        4) Return sorted list of (text, score, id)
        """
        key = f"q::{query}"
        cached = self._cache_get(key)
        if cached:
            return cached

        q_emb = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        D, I = self.index.search(q_emb, top_k*3)  # retrieve more candidates for rerank
        candidate_idxs = I[0].tolist()
        candidates = [(idx, self.texts[idx]) for idx in candidate_idxs if idx < len(self.texts)]
        # BM25 re-rank
        bm25_scores = self.bm25.get_scores(query.split())
        reranked = []
        for idx, text in candidates:
            score = bm25_scores[idx]
            reranked.append({"id": self.metas[idx]["id"], "text": text, "bm25": float(score), "faiss_score": float(D[0][candidate_idxs.index(idx)] if idx in candidate_idxs else 0)})
        # sort by combined score (weighted)
        reranked.sort(key=lambda x: (x["bm25"]*0.6 + x["faiss_score"]*0.4), reverse=True)
        top = reranked[:top_k]
        self._cache_set(key, top)
        return top

    def close(self):
        self.cache.close()

if __name__ == "__main__":
    r = Retriever()
    q = "What is the maternity leave policy?"
    res = r.retrieve(q)
    for r_ in res:
        print(r_["id"], r_["bm25"], r_["text"][:200])
    r.close()

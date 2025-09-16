# app/utils.py
from typing import List, Dict

def chunk_text(text: str, chunk_size: int = 300, chunk_overlap: int = 30) -> List[Dict]:
    """
    Memory-efficient chunker: splits text into overlapping word chunks.
    Defaults to smaller chunk_size to reduce memory footprint.
    """
    words = text.split()
    n = len(words)
    chunks = []
    cid = 0
    i = 0
    while i < n:
        j = min(i + chunk_size, n)
        chunks.append({
            "id": f"chunk_{cid}",
            "text": " ".join(words[i:j]),
            "start": i,
            "end": j
        })
        cid += 1
        # move forward by chunk_size - overlap
        i += (chunk_size - chunk_overlap)
    return chunks

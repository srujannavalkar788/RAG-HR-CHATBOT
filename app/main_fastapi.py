# app/main_fastapi.py
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.retriever import Retriever
from app.llm_client import call_groq_complete
import os
from typing import List, Dict


load_dotenv() 
app = FastAPI(title="RAG HR Chatbot API")
retriever = Retriever()

class QueryIn(BaseModel):
    query: str
    top_k: int = int(os.getenv("TOP_K", 5))

class SourceOut(BaseModel):
    id: str
    text: str
    bm25: float
    faiss_score: float

class QueryOut(BaseModel):
    answer: str
    sources: List[SourceOut]

def build_prompt(query: str, sources: List[Dict]) -> str:
    header = "You are a helpful assistant specialized in HR policies. Use the following source excerpts to answer the question.\n\n"
    src_texts = "\n\n".join([f"Source {i+1}:\n{src['text']}" for i, src in enumerate(sources)])
    prompt = f"{header}{src_texts}\n\nQuestion: {query}\n\nAnswer concisely and cite source numbers where appropriate (e.g., [Source 1])."
    return prompt

@app.post("/query", response_model=QueryOut)
def query(qin: QueryIn):
    try:
        print(f"[DEBUG] Incoming query: {qin.query}")
        candidates = retriever.retrieve(qin.query, top_k=qin.top_k)
        print(f"[DEBUG] Retrieved {len(candidates)} chunks")
        if not candidates:
            raise HTTPException(status_code=404, detail="No relevant sources found.")
        prompt = build_prompt(qin.query, candidates)
        print(f"[DEBUG] Prompt length: {len(prompt)}")
        answer = call_groq_complete(prompt, max_tokens=500)
        print(f"[DEBUG] Answer preview: {answer[:200]}")
        return {"answer": answer, "sources": candidates}
    except Exception as e:
        import traceback
        traceback.print_exc()  # ✅ This will print the full error stack trace
        raise HTTPException(status_code=500, detail=f"Internal Error: {e}")

@app.on_event("shutdown")
def shutdown_event():
    retriever.close()

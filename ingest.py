# app/ingest.py
import os
from PyPDF2 import PdfReader
import json
from pathlib import Path
import re

SRC = Path(__file__).resolve().parents[1] / "app" / "data"
SRC.mkdir(parents=True, exist_ok=True)
PDF_PATH = SRC / "HR_Policy.pdf"  # copy your pdf to this path

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(str(pdf_path))
    texts = []
    for p in reader.pages:
        try:
            texts.append(p.extract_text() or "")
        except Exception:
            texts.append("")
    return "\n".join(texts)

def clean_text(text):
    # simple cleaning: remove repeated whitespace, weird chars, multiple newlines
    t = re.sub(r'\r', '\n', text)
    t = re.sub(r'\n{3,}', '\n\n', t)
    t = re.sub(r'[ \t]+', ' ', t)
    t = t.strip()
    return t

def main(pdf_path=PDF_PATH):
    raw = extract_text_from_pdf(pdf_path)
    cleaned = clean_text(raw)
    out = SRC / "hr_policy_raw.txt"
    out.write_text(cleaned, encoding="utf-8")
    print(f"Saved cleaned text to {out}")
    return out

if __name__ == "__main__":
    main()

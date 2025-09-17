# app/llm_client.py
from dotenv import load_dotenv
import os
import requests
from typing import Any, Dict

#GROQ_API_KEY = os.getenv("GROQ_API_KEY")
#GROQ_API_URL = os.getenv("GROQ_API_URL", "https://console.groq.com/home?utm_source=website&utm_medium=outbound_link&utm_campaign=dev_console_click&_gl=1*vozgl6*_gcl_au*MTc1NDQ2MTIzMS4xNzU3NzMwMTQz*_ga*MTc0NzUyNTY4Ny4xNzU3NzMwMTQy*_ga_4TD0X2GEZG*czE3NTc3MzAxNDEkbzEkZzAkdDE3NTc3MzAxNDEkajYwJGwwJGgw")  # placeholder
#DEFAULT_MODEL = "llama-3.1-8b-instant"  # replace with the correct model name for Groq
load_dotenv()
GROQ_api_KEY = os.getenv("GROQ_API_KEY").strip()
MODEL_NAME = "llama-3.1-8b-instant"

def call_groq_complete(prompt: str, max_tokens: int = 512, temperature: float = 0.0) -> str:
    if not GROQ_api_KEY:
        print("[WARN] No GROQ_API_KEY found. Returning mock response.")
        return "[MOCK RESPONSE] No API key found. Here is a dummy answer."

    url="https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_api_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are an HR assistant. Answer based only on the provided HR policy text."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": temperature
    }

    response = requests.post(url, headers=headers, json=payload)

    # 🔧 Print debug info before parsing
    print(f"[DEBUG] Groq status: {response.status_code}")
    print(f"[DEBUG] Raw response: {response.text[:300]}")  # Show first 300 chars

    if response.status_code != 200:
        return f"[ERROR RESPONSE] Groq API failed (status {response.status_code}): {response.text[:200]}"

    try:
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"[ERROR] Failed to parse Groq response: {e}")
        return "[ERROR RESPONSE] Could not parse model output."
import os
import atexit
import requests
import streamlit as st
from dotenv import load_dotenv
from retriever import Retriever

# ===========================
# Page Configuration (MUST be first Streamlit command)
# ===========================
st.set_page_config(page_title="RAG HR Chatbot", layout="wide")

# ===========================
# Load Environment Variables
# ===========================
load_dotenv()
GROQ_api_KEY = os.getenv("GROQ_API_KEY", "").strip()
API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL_NAME = "llama-3.1-8b-instant"

# ===========================
# Header Section
# ===========================
st.markdown("""
<div style="background-color:#4B79A1; padding:20px; border-radius:15px;">
    <h1 style="color:white; text-align:center;">🤖 RAG HR Chatbot</h1>
    <p style="color:white; text-align:center;">Ask questions about HR policies and get accurate answers with cited sources.</p>
</div>
""", unsafe_allow_html=True)

# ===========================
# Sidebar - Settings
# ===========================
with st.sidebar:
    st.title("⚙️ Settings")
    top_k = st.slider("Number of chunks to retrieve", min_value=1, max_value=10, value=5)
    temperature = st.slider("LLM Temperature", min_value=0.0, max_value=1.0, value=0.0)
    max_tokens = st.slider("Max tokens for LLM", min_value=50, max_value=1024, value=512)

# ===========================
# Session State
# ===========================
if "history" not in st.session_state:
    st.session_state.history = []

# ===========================
# Initialize Retriever
# ===========================
retriever = Retriever()
atexit.register(retriever.close)

# ===========================
# Helper: Call Groq LLM
# ===========================
def call_groq_api(prompt: str):
    if not GROQ_api_KEY:
        return "[ERROR] No API key found."

    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    headers = {"Authorization": f"Bearer {GROQ_api_KEY}", "Content-Type": "application/json"}

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"[ERROR] {str(e)}"

# ===========================
# Display Chat Bubbles
# ===========================
def display_chat_entry(q, a, sources, user_avatar="👤", bot_avatar="🤖"):
    st.markdown(f"""
    <div style="display:flex; justify-content:flex-end; margin:10px;">
        <span style="font-size:25px;">{user_avatar}</span>
        <div style="background-color:blue; padding:12px; border-radius:15px; margin-left:5px; max-width:70%; color:white;">
            {q}
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="display:flex; justify-content:flex-start; margin:10px;">
        <span style="font-size:25px;">{bot_avatar}</span>
        <div style="background-color:blue; color:white; padding:12px; border-radius:15px; margin-left:5px; max-width:70%;">
            {a}
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("📄 Sources"):
        for s in sources:
            st.markdown(f"- **ID:** {s['id']} | BM25: {s['bm25']:.2f} | FAISS: {s['faiss_score']:.2f}")
            st.markdown(f"    > {s['text'][:400]}...")

# ===========================
# Input Box + Button
# ===========================
query = st.text_input("💬 Your question here:", "")

if st.button("Ask") and query.strip():
    with st.spinner("Retrieving relevant HR policy text and generating answer..."):
        sources = retriever.retrieve(query, top_k=top_k)
        context_text = "\n\n".join([f"[Source {i+1}] {s['text']}" for i, s in enumerate(sources)])
        prompt = f"""
You are an HR assistant. Answer ONLY using the following HR policy text.
- Do NOT guess or add information.
- If the information is not present, respond exactly: "The HR policy does not provide information on this question."
- Cite sources like [Source 1], [Source 2] for statements.

HR Policy Text:
{context_text}

Question: {query}
Answer with cited sources:
"""
        answer = call_groq_api(prompt)
        st.session_state.history.append({"q": query, "a": answer, "sources": sources})

# ===========================
# Display Conversation History
# ===========================
for entry in reversed(st.session_state.history):
    display_chat_entry(entry["q"], entry["a"], entry["sources"])

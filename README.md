# 🤖 RAG HR Chatbot

A **Retrieval-Augmented Generation (RAG) powered HR Chatbot** that answers employees' questions about HR policies with **accurate, source-cited responses**.  
Built with **Streamlit**, **Sentence Transformers**, **FAISS**, and **Groq LLMs**.

---

## 📌 Features
- 🔍 **Semantic Search + BM25** – Retrieves the most relevant HR policy chunks.
- 🧠 **RAG Pipeline** – Combines context with LLM for grounded answers.
- 📚 **Source Citations** – Displays the source text, so answers are verifiable.
- 🖥️ **Streamlit UI** – Modern, responsive chat-like interface.
- 🐳 **Dockerized** – Easy to deploy anywhere.
- ⚡ **Caching** – Faster responses with a persistent query cache.

---

## 🏗️ Tech Stack
- **Backend:** Python, FAISS, BM25, SentenceTransformers
- **Frontend:** Streamlit
- **LLM:** Groq (Llama 3.1-8B)
- **Containerization:** Docker

---

## 🚀 Getting Started

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/rag-hr-chatbot.git
cd rag-hr-chatbot
2️⃣ Create Virtual Environment (Optional, for local dev)
python -m venv .venv
source .venv/bin/activate    # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

3️⃣ Add Environment Variables

Create a .env file in the root directory:

GROQ_API_KEY=your_groq_api_key
EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
TOP_K=5
CACHE_TTL=3600

🐳 Run with Docker
1️⃣ Build Image
docker build -t rag-hr-chatbot .

2️⃣ Run Container
docker run -it -p 8501:8501 rag-hr-chatbot


Visit your chatbot at:
👉 http://localhost:8501

🔄 Development Mode (Hot Reload)

For active development, mount the code as a volume:

docker run -it \
  -p 8501:8501 \
  -v ${PWD}:/app \
  rag-hr-chatbot


Now, any local code changes will auto-refresh inside the container.

📁 Project Structure
rag-hr-chatbot/
├── app/
│   ├── data/                 # HR policy text + metadata
│   ├── retriever.py          # FAISS + BM25 retriever
│   ├── streamlit_app.py      # Streamlit UI + RAG pipeline
├── requirements.txt
├── Dockerfile
├── .env.example
└── README.md

🛡️ Security Notes

Do not commit your .env file (contains API keys).

Rotate API keys periodically.

Use limited-scope keys when possible.

🧑‍💻 Contributing

Contributions are welcome!

Fork the repo

Create a feature branch (feature/new-ui)

Commit your changes

Open a pull request 🚀

📜 License

This project is licensed under the MIT License – free to use and modify.

📸 Screenshot

🌐 Deployment

You can deploy this chatbot on:

Streamlit Cloud

Render

AWS / Azure / GCP

Docker + Any Cloud VM

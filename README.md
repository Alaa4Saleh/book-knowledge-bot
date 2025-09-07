# Book Knowledge Bot – AI-Powered RAG over PDFs

Book Knowledge Bot is an **AI-powered Retrieval-Augmented Generation (RAG)** system that processes PDF books, builds embeddings (FAISS + BM25), and provides semantic answers through Fusion and Agent modes with a local Mistral LLM (via Ollama).  
Built with **Streamlit** for an interactive UI + a CLI script for quick testing.

---

## 🚀 Features
- Process any PDF into text chunks with embeddings (`all-MiniLM-L6-v2`)
- Search using **BM25**, **Vector (FAISS)**, or **Hybrid Fusion**
- Agent-RAG mode that dynamically selects the best retrieval strategy
- Evaluation panel: faithfulness, relevancy, context precision, correctness
- Interactive Streamlit web app + CLI retriever for quick demos

---

## 📂 Project Structure
- `app_streamlit.py` – Streamlit app (UI, Fusion/Vector/BM25, Agent-RAG, evaluation)
- `rag_cli.py` – Command-line retriever (BM25 / Vector / Fusion)
- `requirements.txt` – project dependencies
- `.env.example` – environment variables (Ollama settings)
- `data/sample.pdf` – small demo file
- `screenshots/` – example UI screenshots
- `README.md`, `.gitignore`

---

## ⚙️ Installation
```bash
git clone https://github.com/Alaa4Saleh/book-knowledge-bot.git
cd book-knowledge-bot
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env

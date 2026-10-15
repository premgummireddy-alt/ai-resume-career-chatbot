# AI Chatbot for Resume Tailoring and Career Guidance

Retrieval-augmented chatbot built with **Python, LangChain, Llama 2, Pinecone, NLTK, and Streamlit**. Upload a resume, paste a job description, and chat for tailored, honest suggestions.

## Features
- RAG with LangChain: Pinecone vector store (falls back to local FAISS without a key)
- Llama 2 served through any OpenAI-compatible endpoint — works with **NVIDIA TensorRT-LLM / NIM** for accelerated inference, or Ollama/vLLM locally
- NLTK preprocessing: normalization, sentence chunking, de-duplication of the knowledge base
- ChatGPT-style chat UI with per-answer latency and thumbs-up/down feedback logged to `data/feedback.jsonl` for iterative improvement

## Run
```bash
pip install -r requirements.txt
cp .env.example .env   # set LLM_BASE_URL / PINECONE_API_KEY
ollama pull llama2     # or point to a TensorRT-LLM endpoint
export $(grep -v '^#' .env | xargs); streamlit run app/streamlit_app.py
```
Add more guidance documents as `.md` files in `data/`.

## Tests
`pytest tests`

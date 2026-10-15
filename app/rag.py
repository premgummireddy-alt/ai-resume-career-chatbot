"""LangChain RAG pipeline: Pinecone if configured, else local FAISS. LLM: Llama 2 via
any OpenAI-compatible endpoint (e.g. TensorRT-LLM / NIM / vLLM / Ollama)."""
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from preprocess import sentence_chunks, dedupe

PROMPT = ChatPromptTemplate.from_template(
    """You are a career coach. Use the guidance context and the user's resume to answer.
Context:\n{context}\n\nResume:\n{resume}\n\nJob description (optional):\n{job}\n\nQuestion: {question}
Give specific, honest, actionable suggestions. Never invent experience the user does not have.""")

def build_store(docs):
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    chunks = dedupe([c for d in docs for c in sentence_chunks(d)])
    if os.getenv("PINECONE_API_KEY"):
        from langchain_pinecone import PineconeVectorStore
        return PineconeVectorStore.from_texts(chunks, emb, index_name=os.getenv("PINECONE_INDEX", "career-guide"))
    from langchain_community.vectorstores import FAISS
    return FAISS.from_texts(chunks, emb)

def get_llm():
    return ChatOpenAI(base_url=os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"),
                      api_key=os.getenv("LLM_API_KEY", "not-needed"),
                      model=os.getenv("LLM_MODEL", "llama2"), temperature=0.3)

def answer(store, question, resume="", job=""):
    docs = store.similarity_search(question + " " + job[:300], k=4)
    chain = PROMPT | get_llm() | StrOutputParser()
    return chain.invoke({"context": "\n".join(d.page_content for d in docs),
                         "resume": resume[:4000], "job": job[:3000], "question": question})

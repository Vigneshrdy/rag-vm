from fastapi import FastAPI
from pydantic import BaseModel
import os
import re
import requests
from typing import Dict, List

from openai import AzureOpenAI
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# ===============================
# Azure OpenAI
# ===============================
client = AzureOpenAI(
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
)
DEPLOYMENT_NAME = os.environ["AZURE_OPENAI_DEPLOYMENT"]

# ===============================
# FastAPI
# ===============================
app = FastAPI(title="Nyaya AI – Conversational Legal Assistant")

class Query(BaseModel):
    session_id: str
    question: str

# ===============================
# Session Memory (per user)
# ===============================
SESSION_MEMORY: Dict[str, List[str]] = {}

def get_memory(session_id: str) -> str:
    return "\n".join(SESSION_MEMORY.get(session_id, [])[-5:])

def save_memory(session_id: str, text: str):
    SESSION_MEMORY.setdefault(session_id, []).append(text)

# ===============================
# Load FAISS
# ===============================
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.load_local(
    "indian_law_index",
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 6, "fetch_k": 20}
)

# ===============================
# Helpers
# ===============================
def rewrite_query(question: str, memory: str) -> str:
    prompt = f"""
Rewrite the user's question into a complete,
explicit Indian legal query.

Use prior conversation ONLY if helpful.

Conversation:
{memory}

User question:
{question}

Rewritten query:
"""
    resp = client.chat.completions.create(
        model=DEPLOYMENT_NAME,
        messages=[{"role": "user", "content": prompt}]
    )
    return resp.choices[0].message.content.strip()


def needs_web_search(q: str) -> bool:
    keywords = ["latest", "recent", "today", "supreme court judgment", "news"]
    return any(k in q.lower() for k in keywords)


def web_search(query: str) -> str:
    url = "https://duckduckgo.com/html/"
    params = {"q": query + " site:indiankanoon.org OR site:scobserver.in"}
    r = requests.post(url, data=params, timeout=10)
    return r.text[:4000]  # raw HTML slice (LLM will extract meaning)

# ===============================
# Endpoint
# ===============================
@app.post("/ask")
def ask(query: Query):
    session_id = query.session_id
    user_q = query.question.strip()

    memory = get_memory(session_id)

    # 1️⃣ Rewrite query (follow-up resolution)
    rewritten = rewrite_query(user_q, memory)

    # 2️⃣ Decide route
    if needs_web_search(rewritten):
        raw_web = web_search(rewritten)

        system_prompt = (
            "You are a legal assistant. "
            "Answer ONLY from the provided web content. "
            "If unclear, say you do not know."
        )

        user_prompt = f"""
WEB CONTENT:
{raw_web}

QUESTION:
{rewritten}
"""

        resp = client.chat.completions.create(
            model=DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )

        answer = resp.choices[0].message.content.strip()
        save_memory(session_id, f"Q: {user_q}\nA: {answer}")

        return {"answer": answer, "sources": ["web_search"]}

    # 3️⃣ RAG route
    docs = retriever.invoke(rewritten)

    if not docs:
        return {"answer": "I do not know.", "sources": []}

    context = "\n\n".join(d.page_content for d in docs)

    system_prompt = (
        "You are an expert Indian legal assistant. "
        "Answer ONLY using the provided legal context. "
        "If the answer is not present, say 'I do not know.'"
    )

    user_prompt = f"""
LEGAL CONTEXT:
{context}

QUESTION:
{rewritten}
"""

    resp = client.chat.completions.create(
        model=DEPLOYMENT_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )

    answer = resp.choices[0].message.content.strip()
    save_memory(session_id, f"Q: {user_q}\nA: {answer}")

    return {
        "answer": answer,
        "sources": [d.metadata.get("source", "dataset") for d in docs]
    }

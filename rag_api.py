from fastapi import FastAPI
from pydantic import BaseModel
import os
import requests
from typing import Dict, List

from openai import AzureOpenAI
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from bs4 import BeautifulSoup

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
# Session Memory
# ===============================
SESSION_MEMORY: Dict[str, List[str]] = {}

def get_memory(session_id: str) -> str:
    return "\n".join(SESSION_MEMORY.get(session_id, [])[-5:])

def save_memory(session_id: str, text: str):
    SESSION_MEMORY.setdefault(session_id, []).append(text)

# ===============================
# FAISS
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
# Rewrite Query
# ===============================
def rewrite_query(question: str, memory: str) -> str:
    prompt = f"""
Rewrite the user's message into a clear Indian legal research query.
If the user greets or chats casually, return it as-is.

Conversation:
{memory}

User:
{question}

Rewritten query:
"""
    resp = client.chat.completions.create(
        model=DEPLOYMENT_NAME,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content.strip()

# ===============================
# Decide Web Search
# ===============================
def needs_web_search(q: str) -> bool:
    keywords = [
        "latest",
        "recent",
        "today",
        "supreme court",
        "judgment",
        "news",
    ]
    return any(k in q.lower() for k in keywords)

# ===============================
# Indian Kanoon Search (WORKING)
# ===============================
def web_search(query: str) -> str:
    print("🔍 INDIAN KANOON SEARCH QUERY:", query)

    url = "https://duckduckgo.com/html/"
    headers = {
        "User-Agent": "Mozilla/5.0"
    }
    data = {
        "q": f"{query} site:indiankanoon.org OR site:scobserver.in"
    }

    r = requests.post(url, data=data, headers=headers, timeout=15)
    soup = BeautifulSoup(r.text, "html.parser")

    snippets = []
    for result in soup.find_all("div", class_="result"):
        text = result.get_text(" ", strip=True)
        if len(text) > 80:
            snippets.append(text)

    extracted = "\n".join(snippets[:6])
    print("📄 EXTRACTED WEB TEXT LENGTH:", len(extracted))

    return extracted

# ===============================
# Main Endpoint
# ===============================
@app.post("/ask")
def ask(query: Query):
    session_id = query.session_id
    user_q = query.question.strip()

    memory = get_memory(session_id)
    rewritten = rewrite_query(user_q, memory)

    print("🧠 REWRITTEN QUERY:", rewritten)

    # Web search path
    if needs_web_search(rewritten):
        web_text = web_search(rewritten)

        if not web_text.strip():
            return {
                "answer": (
                    "I searched recent Indian legal sources, "
                    "but couldn’t find a clear or authoritative answer yet. "
                    "You may try rephrasing or asking about a specific case."
                ),
                "sources": ["web"]
            }

        system_prompt = (
            "You are Nyaya AI, a friendly and reliable Indian legal assistant.\n"
            "Answer ONLY using the web content provided.\n"
            "Explain clearly in simple language.\n"
            "If the content does not answer the question, say so politely."
        )

        user_prompt = f"""
WEB CONTENT:
{web_text}

QUESTION:
{rewritten}
"""

        resp = client.chat.completions.create(
            model=DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        answer = resp.choices[0].message.content.strip()
        save_memory(session_id, f"Q: {user_q}\nA: {answer}")

        return {
            "answer": answer,
            "sources": ["indiankanoon.org", "scobserver.in"]
        }

    # RAG path
    docs = retriever.invoke(rewritten)

    if not docs:
        return {
            "answer": (
                "I couldn’t find this in my legal database yet. "
                "Try asking about a specific law, section, or act."
            ),
            "sources": []
        }

    context = "\n\n".join(d.page_content for d in docs)

    system_prompt = (
        "You are Nyaya AI, a friendly Indian legal assistant.\n"
        "Answer clearly and accurately using ONLY the legal context.\n"
        "If the answer is missing, say so politely."
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
        ],
    )

    answer = resp.choices[0].message.content.strip()
    save_memory(session_id, f"Q: {user_q}\nA: {answer}")

    return {
        "answer": answer,
        "sources": [d.metadata.get("source", "dataset") for d in docs],
    }

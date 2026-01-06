from fastapi import FastAPI
from pydantic import BaseModel
import os
import re

from openai import AzureOpenAI
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# -----------------------
# Azure OpenAI Client
# -----------------------
client = AzureOpenAI(
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
)

DEPLOYMENT_NAME = os.environ["AZURE_OPENAI_DEPLOYMENT"]

# -----------------------
# FastAPI App
# -----------------------
app = FastAPI(title="Nyaya AI – Indian Law Assistant")

class Query(BaseModel):
    question: str

# -----------------------
# Load Embeddings + FAISS
# -----------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.load_local(
    "indian_law_index",
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# -----------------------
# Helpers
# -----------------------
def is_small_talk(text: str) -> bool:
    patterns = [
        r"\bhi\b",
        r"\bhello\b",
        r"\bhey\b",
        r"how are you",
        r"who are you",
        r"what can you do"
    ]
    text = text.lower()
    return any(re.search(p, text) for p in patterns)


def is_legal_query(question: str) -> bool:
    """
    Azure-safe LLM-based classifier
    (NO temperature parameter)
    """
    prompt = f"""
You are a strict classifier.

Is the following question related to Indian law,
legal rights, FIR, IPC, CrPC, Constitution,
courts, police, or legal procedures?

Answer ONLY with YES or NO.

Question:
{question}
"""

    resp = client.chat.completions.create(
        model=DEPLOYMENT_NAME,
        messages=[{"role": "user", "content": prompt}]
    )

    answer = resp.choices[0].message.content.strip().upper()
    return "YES" in answer

# -----------------------
# API Endpoint
# -----------------------
@app.post("/ask")
def ask(query: Query):
    question = query.question.strip()

    # 1️⃣ Small talk
    if is_small_talk(question):
        return {
            "answer": (
                "Hello! 👋 I’m **Nyaya AI**, an assistant for Indian law. "
                "You can ask me about FIRs, IPC sections, legal procedures, "
                "and your rights in India."
            ),
            "sources": []
        }

    # 2️⃣ Non-legal queries
    if not is_legal_query(question):
        return {
            "answer": (
                "I specialize in Indian legal questions. "
                "Please ask something related to Indian law."
            ),
            "sources": []
        }

    # 3️⃣ Legal RAG
    docs = retriever.invoke(question)

    if not docs:
        return {
            "answer": "I do not know.",
            "sources": []
        }

    context = "\n\n".join(doc.page_content for doc in docs)

    system_prompt = (
        "You are an expert Indian legal assistant. "
        "Answer strictly using ONLY the provided legal context. "
        "If the answer is not found in the context, say 'I do not know.' "
        "Do not guess or add external information."
    )

    user_prompt = f"""
LEGAL CONTEXT:
{context}

QUESTION:
{question}
"""

    response = client.chat.completions.create(
        model=DEPLOYMENT_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        # ❌ No temperature (Azure-safe)
    )

    return {
        "answer": response.choices[0].message.content.strip(),
        "sources": [doc.metadata.get("source", "dataset") for doc in docs]
    }

from fastapi import FastAPI
from pydantic import BaseModel
import os

from google import genai
from google.genai import types

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# -----------------------
# Gemini (NEW SDK)
# -----------------------
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

MODEL_NAME = "gemini-3-flash-preview"   # ✅ supported here

# -----------------------
# App
# -----------------------
app = FastAPI(title="Indian Law RAG")

class Query(BaseModel):
    question: str

# -----------------------
# Load embeddings + FAISS
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
# Endpoint
# -----------------------
@app.post("/ask")
def ask(query: Query):
    docs = retriever.invoke(query.question)

    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = f"""
You are an expert Indian legal assistant.
Answer strictly using the context below.
If the answer is not present, say "I do not know."

CONTEXT:
{context}

QUESTION:
{query.question}
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2
        )
    )

    return {
        "answer": response.text,
        "sources": [doc.metadata.get("source", "dataset") for doc in docs]
    }

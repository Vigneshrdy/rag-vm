from fastapi import FastAPI
from pydantic import BaseModel
import os

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
# App
# -----------------------
app = FastAPI(title="Indian Law RAG (Azure OpenAI)")

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

    system_prompt = (
        "You are an expert Indian legal assistant. "
        "Answer strictly using the provided context. "
        "If the answer is not present, say 'I do not know.'"
    )

    user_prompt = f"""
CONTEXT:
{context}

QUESTION:
{query.question}
"""

    response = client.chat.completions.create(
        model=DEPLOYMENT_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        
    )

    return {
        "answer": response.choices[0].message.content,
        "sources": [doc.metadata.get("source", "dataset") for doc in docs]
    }
